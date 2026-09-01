# src/audit_rasters.py — Phase 2.1: raster EDA ("df.describe()" for maps)
import os
import numpy as np
import rasterio

LAYERS = {
    "dem":       r"data\raw\srtm\srtm_jk_dem.tif",
    "rain_norm": r"data\raw\chirps\chirps_climate_normals.tif",
    "rain_2014": r"data\raw\chirps\chirps_daily_2014.tif",
    "landcover": r"data\raw\landcover\esa_worldcover_jk.tif",
    "soil_sand": r"data\raw\soil\soilgrids_sand_5-15cm.tif",
    "soil_clay": r"data\raw\soil\soilgrids_clay_5-15cm.tif",
    "soil_cfvo": r"data\raw\soil\soilgrids_cfvo_5-15cm.tif",
    "snow":      r"data\raw\snow\modis_snow_final.tif",
    "runoff":    r"data\raw\runoff\era5_runoff_2014.tif",
}

print("=== 1. FILE CHECK (never trust, always verify) ===")
missing = [n for n, p in LAYERS.items() if not os.path.exists(p)]
for name, path in LAYERS.items():
    print(f"{'OK     ' if os.path.exists(path) else 'MISSING'}  {name:10s}  {path}")
if missing:
    raise SystemExit(f"STOP — missing: {missing}")

print("\n=== 2. RASTER AUDIT ===")
for name, path in LAYERS.items():
    with rasterio.open(path) as src:
        b1 = src.read(1).astype("float32")      # band 1 as floats
        valid = b1[np.isfinite(b1)]             # drop NaN/Inf (the raster NaN-handling idiom)
        print(f"\n[{name}]")
        print(f"  CRS        : {src.crs}")
        print(f"  resolution : {tuple(round(r, 5) for r in src.res)}")
        print(f"  bounds     : left={src.bounds.left:.2f} bottom={src.bounds.bottom:.2f} "
              f"right={src.bounds.right:.2f} top={src.bounds.top:.2f}")
        print(f"  bands={src.count}  size={src.width}x{src.height}  dtype={src.dtypes[0]}  nodata={src.nodata}")
        if valid.size:
            print(f"  band-1 stats: min={valid.min():.3f}  median={np.median(valid):.3f}  max={valid.max():.3f}")
        else:
            print("  band-1 stats: ALL INVALID — investigate!")