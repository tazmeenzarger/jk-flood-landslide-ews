# src/generate_flood_labels.py — Path A Update
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio

# UPDATED: Pointing to the Global Flood Database composite
FLOOD_RASTER = r"data\raw\flood_extent\gfd_jk_composite.tif"
OUT_PATH = r"data\processed\labels_flood.gpkg"

print("Loading GFD flood extent raster...")
with rasterio.open(FLOOD_RASTER) as src:
    data = src.read(1)
    transform = src.transform
    crs = src.crs

flood_rows, flood_cols = np.where(data == 1)
bg_rows, bg_cols = np.where(data == 0)
print(f"Total flooded pixels available: {len(flood_rows)}")

np.random.seed(42)
n_flood_sample = min(1000, len(flood_rows))
flood_idx = np.random.choice(len(flood_rows), n_flood_sample, replace=False)
f_rows = flood_rows[flood_idx].tolist()
f_cols = flood_cols[flood_idx].tolist()

n_bg_sample = min(n_flood_sample * 3, len(bg_rows))
bg_idx = np.random.choice(len(bg_rows), n_bg_sample, replace=False)
b_rows = bg_rows[bg_idx].tolist()
b_cols = bg_cols[bg_idx].tolist()

flood_xs, flood_ys = rasterio.transform.xy(transform, f_rows, f_cols)
bg_xs, bg_ys = rasterio.transform.xy(transform, b_rows, b_cols)

all_xs = list(flood_xs) + list(bg_xs)
all_ys = list(flood_ys) + list(bg_ys)
all_labels = [1]*len(flood_xs) + [0]*len(bg_xs)

combined = gpd.GeoDataFrame({'is_flood': all_labels}, geometry=[Point(x, y) for x, y in zip(all_xs, all_ys)], crs=crs)
combined = combined.to_crs("EPSG:32643")
combined.to_file(OUT_PATH, driver="GPKG")
print(f"Saved {len(flood_xs)} flood / {len(bg_xs)} bg to {OUT_PATH}")