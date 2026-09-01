# src/define_target_grid.py — Phase 2.2: The Target Grid Blueprint
import os
import json
import rasterio
from rasterio.warp import transform_bounds
import math

# 1. Configuration
# We use the DEM as our reference for the "True" extent of J&K
REF_PATH = r"data\raw\srtm\srtm_jk_dem.tif"
TARGET_CRS = "EPSG:32643"  # UTM Zone 43N (Meters)
TARGET_RES = 1000.0        # 1 km resolution
OUTPUT_DIR = r"data\processed"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "target_grid.json")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Read Reference Extent
with rasterio.open(REF_PATH) as src:
    # Get bounds in degrees (EPSG:4326)
    bounds_4326 = src.bounds
    print(f"Reference bounds (deg): {bounds_4326}")
    
    # Transform bounds to Target CRS (Meters)
    # We use a densification factor (21) to ensure the curved edges of the bbox
    # are transformed accurately, not just the 4 corners.
    bounds_target = transform_bounds(src.crs, TARGET_CRS, *bounds_4326, densify_pts=21)
    print(f"Transformed bounds (m) : {bounds_target}")

# 3. Snap to Grid (The "Cookie Cutter" Math)
# We want our pixels to align perfectly with the 1000m grid.
# left: round DOWN to nearest 1000
# bottom: round DOWN to nearest 1000
# right: round UP to nearest 1000
# top: round UP to nearest 1000
left   = math.floor(bounds_target[0] / TARGET_RES) * TARGET_RES
bottom = math.floor(bounds_target[1] / TARGET_RES) * TARGET_RES
right  = math.ceil(bounds_target[2] / TARGET_RES)  * TARGET_RES
top    = math.ceil(bounds_target[3] / TARGET_RES)  * TARGET_RES

# Calculate dimensions
width  = int((right - left) / TARGET_RES)
height = int((top - bottom) / TARGET_RES)

print(f"\nTARGET GRID DEFINITION:")
print(f"  CRS      : {TARGET_CRS}")
print(f"  Res      : {TARGET_RES} m")
print(f"  Bounds   : ({left}, {bottom}, {right}, {top})")
print(f"  Size     : {width} x {height} pixels")

# 4. Save the Blueprint
grid_info = {
    "crs": TARGET_CRS,
    "res": TARGET_RES,
    "bounds": (left, bottom, right, top),
    "width": width,
    "height": height,
    "ref_path": REF_PATH
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(grid_info, f, indent=4)

print(f"\nBlueprint saved to: {OUTPUT_JSON}")
print("This JSON file will now drive all future alignment scripts.")