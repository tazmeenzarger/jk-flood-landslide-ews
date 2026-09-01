# src/fetch_dfo_inventory.py — Phase 4.1 (v5): Robust Spatial GPKG Inventory
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

GPKG_PATH = r"data\raw\Global_Flood_Records.gpkg"

print("Loading Global Flood Records GeoPackage...")
gdf = gpd.read_file(GPKG_PATH)
print(f"Loaded {len(gdf)} global spatial records.")
print(f"Available columns: {gdf.columns.tolist()}")

# Helper: find first column whose lowercase name contains any of the keys
def find_col(frame, keys):
    for c in frame.columns:
        cl = str(c).lower()
        for k in keys:
            if k in cl:
                return c
    return None

start_col   = find_col(gdf, ['began', 'start', 'begin'])
end_col     = find_col(gdf, ['ended', 'end'])
country_col = find_col(gdf, ['country'])
cause_col   = find_col(gdf, ['cause'])
sev_col     = find_col(gdf, ['severity'])

print(f"Detected -> start:{start_col} | end:{end_col} | country:{country_col} | cause:{cause_col} | severity:{sev_col}")

# J&K bounding box (Lon: 73-80.5, Lat: 32-37.5)
jk_poly = box(73.0, 32.0, 80.5, 37.5)
jk_floods = gdf[gdf.intersects(jk_poly)].copy()
print(f"\nFound {len(jk_floods)} historical flood events intersecting J&K!")

if start_col:
    jk_floods['Began'] = pd.to_datetime(jk_floods[start_col], errors='coerce')
    if end_col:
        jk_floods['Ended'] = pd.to_datetime(jk_floods[end_col], errors='coerce')
    jk_floods = jk_floods.dropna(subset=['Began'])

    print("\n=== J&K FLOOD HISTORY (2000+) ===")
    cols_to_print = ['Began']
    if end_col: cols_to_print.append('Ended')
    for c in [country_col, cause_col, sev_col]:
        if c: cols_to_print.append(c)
    recent = jk_floods[jk_floods['Began'] >= '2000-01-01']
    print(recent[cols_to_print].to_string())

    out_cols = ['Began'] + (['Ended'] if end_col else []) + [c for c in [country_col, cause_col, sev_col] if c]
    jk_floods[out_cols].to_csv(r"data\processed\j_k_flood_inventory.csv", index=False)
    print("\nSaved filtered inventory to: data\\processed\\j_k_flood_inventory.csv")
else:
    print("ERROR: Could not detect a start-date column. See the column list printed above.")