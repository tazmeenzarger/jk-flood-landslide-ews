# src/align_rasters.py — Phase 2.4: The Great Alignment (v2 - dtype safe)
import os
import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

# 1. Load the Blueprint
with open(r"data\processed\target_grid.json", "r") as f:
    grid = json.load(f)

TARGET_CRS = grid["crs"]
TARGET_BOUNDS = grid["bounds"]
TARGET_WIDTH = grid["width"]
TARGET_HEIGHT = grid["height"]

target_transform = from_bounds(*TARGET_BOUNDS, TARGET_WIDTH, TARGET_HEIGHT)

OUT_DIR = r"data\processed\aligned"
os.makedirs(OUT_DIR, exist_ok=True)

# 2. Define the Pipeline: (source_path, out_name, resampling_method, scale_factor)
LAYERS = [
    (r"data\raw\srtm\srtm_jk_dem.tif", "dem.tif", Resampling.average, 1.0),
    (r"data\processed\terrain\slope_utm30.tif", "slope.tif", Resampling.average, 1.0),
    (r"data\processed\terrain\twi_utm30.tif", "twi.tif", Resampling.average, 1.0),
    (r"data\raw\chirps\chirps_climate_normals.tif", "rain_norm.tif", Resampling.bilinear, 1.0),
    (r"data\raw\chirps\chirps_daily_2014.tif", "rain_2014.tif", Resampling.bilinear, 1.0),
    (r"data\raw\landcover\esa_worldcover_jk.tif", "landcover.tif", Resampling.nearest, 1.0),
    (r"data\raw\soil\soilgrids_sand_5-15cm.tif", "soil_sand.tif", Resampling.average, 0.1), # Fix x10 scale
    (r"data\raw\soil\soilgrids_clay_5-15cm.tif", "soil_clay.tif", Resampling.average, 0.1),
    (r"data\raw\soil\soilgrids_cfvo_5-15cm.tif", "soil_cfvo.tif", Resampling.average, 0.1),
    (r"data\raw\snow\modis_snow_final.tif", "snow.tif", Resampling.bilinear, 1.0),
    (r"data\raw\runoff\era5_runoff_2014.tif", "runoff.tif", Resampling.bilinear, 1.0),
]

# 3. The Alignment Loop
for src_path, out_name, resamp, scale in LAYERS:
    out_path = os.path.join(OUT_DIR, out_name)
    print(f"Aligning {out_name}...")
    
    with rasterio.open(src_path) as src:
        # --- ROBUST NODATA HANDLER ---
        dtype = np.dtype(src.dtypes[0])
        if src.nodata is not None:
            src_nodata = src.nodata
        elif np.issubdtype(dtype, np.floating):
            src_nodata = -9999.0
        elif np.issubdtype(dtype, np.integer):
            # If unsigned (uint8), use the max value (255). If signed (int16), use the min value (-32768).
            src_nodata = np.iinfo(dtype).max if dtype.kind == 'u' else np.iinfo(dtype).min
        else:
            src_nodata = -9999.0
            
        profile = src.profile.copy()
        profile.update(
            crs=TARGET_CRS,
            transform=target_transform,
            width=TARGET_WIDTH,
            height=TARGET_HEIGHT,
            nodata=src_nodata,
            compress="lzw" # Keep file sizes small
        )
        
        with rasterio.open(out_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=target_transform,
                    dst_crs=TARGET_CRS,
                    dst_nodata=src_nodata,
                    resampling=resamp
                )
                
        # Apply the SoilGrids scale factor (and safely ignore NoData pixels)
        if scale != 1.0:
            with rasterio.open(out_path, "r+") as dst:
                for i in range(1, dst.count + 1):
                    data = dst.read(i).astype("float32")
                    mask = (data != src_nodata)
                    data[mask] = data[mask] * scale
                    dst.write(data, i)
                    
print("ALL LAYERS ALIGNED!")