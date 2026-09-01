# src/build_regional_flood_inventories.py — Phase 8.1a: Four-region DFO inventories
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

GPKG = r"data\raw\Global_Flood_Records.gpkg"
REGIONS = {
    'jk':          box(73.0, 32.0, 80.5, 37.5),
    'himachal':    box(75.5, 30.5, 79.5, 33.5),
    'uttarakhand': box(77.5, 28.5, 81.5, 31.5),
    'nepal':       box(80.0, 26.5, 88.5, 30.5),
}

gdf = gpd.read_file(GPKG)
for name, poly in REGIONS.items():
    sub = gdf[gdf.intersects(poly)].copy()
    sub['Began'] = pd.to_datetime(sub['BeginDate'], errors='coerce')
    sub['Ended'] = pd.to_datetime(sub['EndDate'], errors='coerce')
    sub = sub.dropna(subset=['Began'])
    out = sub[['Began', 'Ended', 'Country', 'MainCause', 'Severity']]
    out.to_csv(rf"data\processed\flood_inventory_{name}.csv", index=False)
    print(f"{name}: {len(out)} events")