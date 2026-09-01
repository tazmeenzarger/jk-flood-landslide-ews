# src/derive_hand.py — Phase 3B Path A: HAND (Height Above Nearest Drainage) - FIXED
import os, json
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from whitebox import WhiteboxTools

TERRAIN_DIR = r"data\processed\terrain"
OUT_DIR = r"data\processed\aligned"
abs_terrain = os.path.abspath(TERRAIN_DIR)

wbt = WhiteboxTools()
wbt.work_dir = abs_terrain
wbt.verbose = True

# 1) Hydrological conditioning (skipped if already exists from last run!)
if not os.path.exists(os.path.join(abs_terrain, "dem_filled.tif")):
    print("Filling depressions...")
    wbt.run_tool("FillDepressions", ["--dem=dem_utm30.tif", "--output=dem_filled.tif", "--fix_flats"])
else:
    print("dem_filled.tif already exists, skipping 5-min fill step.")

# 2) D8 Flow Accumulation (Counting upstream cells)
if not os.path.exists(os.path.join(abs_terrain, "d8fa.tif")):
    print("Computing D8 Flow Accumulation...")
    wbt.run_tool("D8FlowAccumulation", ["--dem=dem_filled.tif", "--output=d8fa.tif", "--out_type=cells"])

# 3) Extract Streams (Threshold of 10000 cells defines a major river in this large basin)
if not os.path.exists(os.path.join(abs_terrain, "streams.tif")):
    print("Extracting stream network...")
    wbt.run_tool("ExtractStreams", ["--flow_accum=d8fa.tif", "--output=streams.tif", "--threshold=10000"])

# 4) HAND (Elevation Above Stream)
if not os.path.exists(os.path.join(abs_terrain, "hand_30m.tif")):
    print("Computing Height Above Nearest Drainage (HAND)...")
    wbt.run_tool("ElevationAboveStream", ["--dem=dem_filled.tif", "--streams=streams.tif", "--output=hand_30m.tif"])

# 5) Align HAND to the 1 km target grid
out_path = os.path.join(OUT_DIR, "hand.tif")
with open(r"data\processed\target_grid.json") as f:
    grid = json.load(f)
target_transform = from_bounds(*grid["bounds"], grid["width"], grid["height"])

with rasterio.open(os.path.join(abs_terrain, "hand_30m.tif")) as src:
    profile = src.profile.copy()
    profile.update(crs=grid["crs"], transform=target_transform,
                   width=grid["width"], height=grid["height"],
                   dtype="float32", nodata=-9999.0, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        reproject(source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                  src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=target_transform, dst_crs=grid["crs"], dst_nodata=-9999.0,
                  resampling=Resampling.average)

# 6) Cleanup the massive intermediate files once HAND is safely aligned
for f in ["dem_filled.tif", "d8fa.tif", "streams.tif"]:
    p = os.path.join(abs_terrain, f)
    if os.path.exists(p) and os.path.exists(out_path):
        os.remove(p)
        print(f"Cleaned {f}")

print("HAND derived and aligned to 1 km.")