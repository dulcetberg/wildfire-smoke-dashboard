import os
import zipfile
import glob
import datetime
import requests
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson, HeatMap
from dotenv import load_dotenv

load_dotenv()
AIRNOW_API_KEY = os.environ["AIRNOW_API_KEY"]

# Rolling window: always "yesterday back 7 days" rather than fixed dates, so a
# scheduled rebuild keeps showing current conditions without manual updates.
# END_DATE is yesterday, not today, since both NOAA HMS and AirNow lag a bit
# behind real time and today's data would otherwise be incomplete.
END_DATE = datetime.date.today() - datetime.timedelta(days=1)
START_DATE = END_DATE - datetime.timedelta(days=7)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Chicago-area bounding box, wide enough to catch nearby IN/WI monitors too
CHI_BBOX = (-88.3, 41.4, -87.3, 42.3)

# Canada + Great Lakes region bounding box for smoke/fire source data
NORTH_BBOX = (-100.0, 40.0, -75.0, 60.0)


def daterange(start, end):
    days = (end - start).days + 1
    for i in range(days):
        yield start + datetime.timedelta(days=i)


def fetch_hms_layer(kind, out_gpkg):
    """kind is 'Smoke_Polygons' or 'Fire_Points'. Downloads + merges daily HMS shapefiles into one GeoPackage with a date column."""
    print(f"\n[HMS] Fetching {kind} for {START_DATE} to {END_DATE}...")
    frames = []
    for day in daterange(START_DATE, END_DATE):
        ymd = day.strftime("%Y%m%d")
        prefix = "hms_smoke" if kind == "Smoke_Polygons" else "hms_fire"
        url = f"https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/{kind}/Shapefile/{day:%Y}/{day:%m}/{prefix}{ymd}.zip"
        local_zip = f"{RAW_DIR}/{prefix}{ymd}.zip"
        if not os.path.exists(local_zip):
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                print(f"  -> {ymd}: not available ({r.status_code}), skipping")
                continue
            with open(local_zip, "wb") as f:
                f.write(r.content)
        extract_dir = f"{RAW_DIR}/{prefix}{ymd}"
        if not os.path.exists(extract_dir):
            with zipfile.ZipFile(local_zip) as z:
                z.extractall(extract_dir)
        shp_files = glob.glob(f"{extract_dir}/*.shp")
        if not shp_files:
            print(f"  -> {ymd}: no shapefile found in archive, skipping")
            continue
        try:
            gdf = gpd.read_file(shp_files[0])
        except Exception as e:
            print(f"  -> {ymd}: failed to read shapefile ({e}), skipping")
            continue
        gdf["date"] = day.isoformat()
        frames.append(gdf)
        print(f"  -> {ymd}: {len(gdf)} features")

    if not frames:
        print(f"  -> No {kind} data retrieved at all.")
        return gpd.GeoDataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)
    combined = combined.to_crs("EPSG:4326")

    # Clip to region of interest to keep the map payload manageable
    minx, miny, maxx, maxy = NORTH_BBOX
    combined = combined.cx[minx:maxx, miny:maxy]

    combined.to_file(f"{PROCESSED_DIR}/{out_gpkg}", driver="GPKG")
    print(f"  -> Saved {len(combined)} clipped features to {PROCESSED_DIR}/{out_gpkg}")
    return combined


def fetch_airnow_pm25():
    print(f"\n[AirNow] Fetching PM2.5 observations for Chicago area, {START_DATE} to {END_DATE}...")
    minx, miny, maxx, maxy = CHI_BBOX
    url = (
        "https://www.airnowapi.org/aq/data/"
        f"?startDate={START_DATE}T00&endDate={END_DATE}T23"
        "&parameters=PM25"
        f"&BBOX={minx},{miny},{maxx},{maxy}"
        "&dataType=B&format=application/json&verbose=1&monitorType=0"
        f"&includerawconcentrations=0&API_KEY={AIRNOW_API_KEY}"
    )
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    records = r.json()
    df = pd.DataFrame(records)
    df["UTC"] = pd.to_datetime(df["UTC"])
    df.to_csv(f"{PROCESSED_DIR}/airnow_pm25.csv", index=False)
    print(f"  -> {len(df)} readings across {df['SiteName'].nunique()} stations saved to {PROCESSED_DIR}/airnow_pm25.csv")
    return df


AQI_COLORS = {
    1: "#00e400",  # Good
    2: "#ffff00",  # Moderate
    3: "#ff7e00",  # Unhealthy for Sensitive Groups
    4: "#ff0000",  # Unhealthy
    5: "#8f3f97",  # Very Unhealthy
    6: "#7e0023",  # Hazardous
}


def build_dashboard(smoke_gdf, fire_gdf, aqi_df, out_html="index.html"):
    print("\n[Dashboard] Building Folium map...")
    m = folium.Map(location=[43.0, -85.0], zoom_start=5, tiles="CartoDB dark_matter")

    # --- Time-animated smoke polygons ---
    if len(smoke_gdf):
        smoke_features = []
        for _, row in smoke_gdf.iterrows():
            if row.geometry is None:
                continue
            smoke_features.append({
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {
                    # Anchored to noon UTC (not midnight) so any US timezone still
                    # displays the correct calendar date in the time slider.
                    "time": row["date"] + "T12:00:00Z",
                    "style": {"color": "#999999", "fillColor": "#999999", "fillOpacity": 0.25, "weight": 0.5},
                },
            })
        TimestampedGeoJson(
            {"type": "FeatureCollection", "features": smoke_features},
            period="P1D",
            duration="P1D",
            transition_time=800,
            auto_play=False,
            loop=False,
            add_last_point=False,
        ).add_to(m)

    # --- Fire detection density, whole-period overview (229k+ raw points is too many for
    # individual markers, and a second time-animated layer conflicts with the smoke
    # TimestampedGeoJson layer's leaflet-timedimension instance, so this is static) ---
    if len(fire_gdf):
        fire_gdf = fire_gdf[fire_gdf.geometry.notnull()]
        # Round to a coarse grid and count, so density shows through as weight
        # rather than rendering every single raw detection point individually.
        coords = list(zip(fire_gdf.geometry.y.round(2), fire_gdf.geometry.x.round(2)))
        counts = pd.Series(coords).value_counts()
        heat_points = [[lat, lon, min(count, 100)] for (lat, lon), count in counts.items()]

        HeatMap(
            heat_points,
            radius=5,
            blur=6,
            max_opacity=0.7,
            gradient={"0.2": "#ffff00", "0.5": "#ff7e00", "0.8": "#ff0000", "1.0": "#7e0023"},
            name="Fire detection density (satellite, whole period)",
        ).add_to(m)

    # --- Chicago-area AQI stations, peak reading per site ---
    if len(aqi_df):
        peak = aqi_df.loc[aqi_df.groupby("SiteName")["AQI"].idxmax()]
        station_layer = folium.FeatureGroup(name="Chicago-area PM2.5 monitors (peak AQI)", show=True)
        for _, row in peak.iterrows():
            color = AQI_COLORS.get(int(row["Category"]), "#808080")
            popup_html = (
                f"<b>{row['SiteName']}</b><br>"
                f"Peak AQI: {int(row['AQI'])}<br>"
                f"PM2.5: {row['Value']} {row['Unit']}<br>"
                f"At: {row['UTC']}"
            )
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=220),
            ).add_to(station_layer)
        station_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    start = START_DATE.strftime("%b %-d")
    end = END_DATE.strftime("%b %-d, %Y")
    title_html = f"""
    <div style="
        position: fixed;
        top: 10px;
        left: 50px;
        z-index: 9999;
        max-width: 340px;
        background: rgba(20, 20, 20, 0.85);
        color: #f2f2f2;
        font-family: -apple-system, Helvetica, Arial, sans-serif;
        padding: 12px 16px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    ">
        <h3 style="margin: 0 0 6px; font-size: 1.05rem; color: #ff9a4d;">
            Canadian Wildfire Smoke — Impact on the Chicago Area
        </h3>
        <p style="margin: 0 0 8px; font-size: 0.82rem; line-height: 1.4;">
            A live, auto-updating look at wildfire smoke reaching the Chicago area: animated
            daily smoke plume extent and satellite fire detections from NOAA's Hazard Mapping
            System, alongside hourly Chicago-area PM2.5/AQI readings from the EPA AirNow API.
            Currently showing {start} &ndash; {end}, refreshed automatically each day.
        </p>
        <p style="margin: 0; font-size: 0.78rem; color: #bbbbbb;">
            By Brian Bergstrom &middot;
            <a href="https://bergstromgis.com" style="color: #ff9a4d;" target="_blank" rel="noopener">bergstromgis.com</a>
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(out_html)
    print(f"\nDashboard saved to {out_html} — open it directly in any browser.")


if __name__ == "__main__":
    smoke = fetch_hms_layer("Smoke_Polygons", "smoke.gpkg")
    fire = fetch_hms_layer("Fire_Points", "fire.gpkg")
    aqi = fetch_airnow_pm25()
    build_dashboard(smoke, fire, aqi)
