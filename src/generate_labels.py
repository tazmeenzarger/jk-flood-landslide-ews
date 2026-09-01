# src/generate_labels.py — Phase 2.5 (v3 — exact match + range validation)
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

BOUNDARY_PATH = r"data\processed\jk_boundary.shp"
CATALOG_PATH  = r"data\raw\nasa_landslide_catalog.csv"
OUT_PATH      = r"data\processed\labels.gpkg"
TARGET_CRS    = "EPSG:32643"

# --- 1. Boundary ---
boundary = gpd.read_file(BOUNDARY_PATH).to_crs(TARGET_CRS)
boundary = boundary.make_valid()
jk_poly = boundary.union_all() if hasattr(boundary, "union_all") else boundary.unary_union
print(f"Boundary: type={jk_poly.geom_type} | valid={jk_poly.is_valid}")

# --- 2. Catalog with EXACT column matching ---
df = pd.read_csv(CATALOG_PATH)
lat_col = next((c for c in df.columns if c.strip().lower() in ("latitude", "lat")), None)
lon_col = next((c for c in df.columns if c.strip().lower() in ("longitude", "lon", "lng", "long")), None)
print(f"Detected lat='{lat_col}' lon='{lon_col}'")
if lat_col is None or lon_col is None:
    raise SystemExit(f"Column detection failed. Columns: {df.columns.tolist()}")

lat = pd.to_numeric(df[lat_col], errors="coerce")
lon = pd.to_numeric(df[lon_col], errors="coerce")

# Keep only physically possible coordinates (defensive parsing)
ok = lat.between(-90, 90) & lon.between(-180, 180)
print(f"Rows: {len(df)} | valid coordinate pairs: {int(ok.sum())}")

events = gpd.GeoDataFrame(
    df.loc[ok], geometry=gpd.points_from_xy(lon[ok], lat[ok]), crs="EPSG:4326"
).to_crs(TARGET_CRS)

inside = events.geometry.within(jk_poly)
print(f"Events inside J&K: {int(inside.sum())}")

if inside.sum() == 0:
    print(f"DIAGNOSTIC — events bounds (UTM): {events.total_bounds}")
    raise SystemExit("Still 0 inside — paste this output.")

events_jk = events[inside].copy()
events_jk["is_landslide"] = 1
print(f"Using {len(events_jk)} landslides.")

# --- 3. Background points (seeded for reproducibility) ---
n_background = len(events_jk) * 3
minx, miny, maxx, maxy = jk_poly.bounds
rng = np.random.default_rng(42)
bg_points = []
while len(bg_points) < n_background:
    rx = rng.uniform(minx, maxx, n_background * 2)
    ry = rng.uniform(miny, maxy, n_background * 2)
    for x, y in zip(rx, ry):
        if jk_poly.contains(Point(x, y)):
            bg_points.append(Point(x, y))
            if len(bg_points) >= n_background:
                break

bg_gdf = gpd.GeoDataFrame({"is_landslide": [0] * len(bg_points)}, geometry=bg_points, crs=TARGET_CRS)

# --- 4. Combine & save ---
combined = pd.concat([events_jk[["geometry", "is_landslide"]], bg_gdf], ignore_index=True)
labels_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs=TARGET_CRS)
labels_gdf.to_file(OUT_PATH, driver="GPKG")

print(f"\n=== LABEL GENERATION COMPLETE ===")
print(f"Total: {len(labels_gdf)} | Landslides: {len(events_jk)} | Background: {len(bg_gdf)}")
print(f"Saved to: {OUT_PATH}")