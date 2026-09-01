# src/derive_terrain.py — Phase 2.3: 30m TWI (v6 - Space Managed)
import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from whitebox import WhiteboxTools

RAW_DEM  = r"data\raw\srtm\srtm_jk_dem.tif"
OUT_DIR  = r"data\processed\terrain"
DST_CRS  = "EPSG:32643"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_DIR_ABS = os.path.abspath(OUT_DIR)

DEM_UTM   = "dem_utm30.tif"
FILLED    = "dem_filled.tif"
SLOPE_DEG = "slope_utm30.tif"
SLOPE_RAD = "slope_rad.tif"
SCA       = "sca.tif"
TWI       = "twi_utm30.tif"

# --- THE JANITOR ---
def is_valid_raster(fname):
    fpath = os.path.join(OUT_DIR_ABS, fname)
    if not os.path.exists(fpath): return False
    try:
        with rasterio.open(fpath) as src:
            _ = src.width
            return True
    except Exception:
        print(f"  [Janitor] Deleting corrupt/invalid file: {fname}")
        os.remove(fpath)
        return False

for f in [FILLED, SLOPE_DEG, SLOPE_RAD, SCA, TWI]:
    is_valid_raster(f)

# --- 1) Reproject DEM ---
if not is_valid_raster(DEM_UTM):
    print("Reprojecting DEM to UTM 43N...")
    with rasterio.open(RAW_DEM) as src:
        transform, width, height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds, resolution=30)
        profile = src.profile.copy()
        profile.update(crs=DST_CRS, transform=transform, width=width, height=height,
                       dtype="float32", nodata=-9999.0, compress="deflate")
        with rasterio.open(os.path.join(OUT_DIR_ABS, DEM_UTM), "w", **profile) as dst:
            reproject(source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=DST_CRS,
                      resampling=Resampling.bilinear)
else:
    print(f"[Skip] {DEM_UTM} is valid.")

wbt = WhiteboxTools()
wbt.work_dir = OUT_DIR_ABS
wbt.verbose = True

# --- 2) Fill Depressions ---
if not is_valid_raster(FILLED):
    print("Filling depressions...")
    wbt.run_tool("FillDepressions", ["--dem=dem_utm30.tif", "--output=dem_filled.tif", "--fix_flats"])
else:
    print(f"[Skip] {FILLED} is valid.")

# --- 3) Slope (Degrees) ---
if not is_valid_raster(SLOPE_DEG):
    print("Computing slope (degrees)...")
    wbt.run_tool("Slope", ["--dem=dem_filled.tif", "--output=slope_utm30.tif", "--units=degrees"])
else:
    print(f"[Skip] {SLOPE_DEG} is valid.")

# --- 4) TWI Pipeline ---
if not is_valid_raster(SLOPE_RAD):
    print("Computing slope (radians) for TWI...")
    wbt.run_tool("Slope", ["--dem=dem_filled.tif", "--output=slope_rad.tif", "--units=radians"])
else:
    print(f"[Skip] {SLOPE_RAD} is valid.")

if not is_valid_raster(SCA):
    print("Computing Specific Catchment Area (SCA) via D8 Flow Accumulation...")
    wbt.run_tool("D8FlowAccumulation", ["--dem=dem_filled.tif", "--output=sca.tif", "--out_type=sca"])
else:
    print(f"[Skip] {SCA} is valid.")

if not is_valid_raster(TWI):
    print("Computing TWI...")
    wbt.run_tool("WetnessIndex", ["--sca=sca.tif", "--slope=slope_rad.tif", "--output=twi_utm30.tif"])
else:
    print(f"[Skip] {TWI} is valid.")

# --- 5) THE CLEANUP CREW (Keeps disk space free for Phase 2.4!) ---
print("\nCleaning up massive intermediate files...")
for f in [FILLED, SLOPE_RAD, SCA]:
    p = os.path.join(OUT_DIR_ABS, f)
    if os.path.exists(p):
        os.remove(p)
        print(f"  Freed space by deleting {f}")

# --- 6) Sanity stats ---
print("\n=== FINAL STATS ===")
for fname in [SLOPE_DEG, TWI]:
    path = os.path.join(OUT_DIR_ABS, fname)
    if os.path.exists(path):
        with rasterio.open(path) as src:
            d = src.read(1).astype("float32")
            d[d == src.nodata] = np.nan
            d[d < -9000] = np.nan
            print(f"{fname}: min={np.nanmin(d):.2f} median={np.nanmedian(d):.2f} max={np.nanmax(d):.2f}")

print("DONE")