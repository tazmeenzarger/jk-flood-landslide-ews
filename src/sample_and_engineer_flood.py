# src/sample_and_engineer_flood.py — Path A Update (Adds HAND)
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol

LABELS_PATH = r"data\processed\labels_flood.gpkg"
ALIGNED_DIR = r"data\processed\aligned"
OUT_PATH = r"data\processed\features_flood_final.csv"

# UPDATED: Added "hand" to the raster dictionary
RASTERS = {
    "dem": "dem.tif", "slope": "slope.tif", "twi": "twi.tif", "hand": "hand.tif",
    "landcover": "landcover.tif",
    "soil_sand": "soil_sand.tif", "soil_clay": "soil_clay.tif", "soil_cfvo": "soil_cfvo.tif",
    "rain_norm": "rain_norm.tif", "rain_2014": "rain_2014.tif",
    "snow": "snow.tif", "runoff": "runoff.tif"
}

print("Loading 4000 flood/background points...")
gdf = gpd.read_file(LABELS_PATH).to_crs("EPSG:32643")
xs = gdf.geometry.x.to_numpy()
ys = gdf.geometry.y.to_numpy()

dfs = [pd.DataFrame({"is_flood": gdf["is_flood"]})]

print("Interrogating aligned rasters (vectorized)...")
for name, fname in RASTERS.items():
    print(f"  Sampling {name}...", end=" ", flush=True)
    with rasterio.open(os.path.join(ALIGNED_DIR, fname)) as src:
        rows, cols = rowcol(src.transform, xs, ys)
        rows = np.clip(np.asarray(rows), 0, src.height - 1)
        cols = np.clip(np.asarray(cols), 0, src.width - 1)

        vals = np.empty((len(rows), src.count), dtype="float32")
        for b in range(src.count):
            vals[:, b] = src.read(b + 1)[rows, cols]

        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan

        if src.count == 1:
            dfs.append(pd.DataFrame({name: vals[:, 0]}))
        else:
            cols_names = [f"{name}_day_{i+1}" if name in ("rain_2014", "runoff")
                          else f"{name}_band_{i+1}" for i in range(src.count)]
            dfs.append(pd.DataFrame(vals, columns=cols_names))
    print("Done.")

print("\nConcatenating and Engineering features...")
df = pd.concat(dfs, axis=1)

rain_cols = [c for c in df.columns if c.startswith("rain_2014_day_")]
runoff_cols = [c for c in df.columns if c.startswith("runoff_day_")]
df[rain_cols + runoff_cols] = df[rain_cols + runoff_cols].fillna(0)

df["rain_event_7d"] = df[[f"rain_2014_day_{d}" for d in range(245, 252)]].sum(axis=1)
df["rain_antecedent_30d"] = df[[f"rain_2014_day_{d}" for d in range(215, 245)]].sum(axis=1)
df["rain_max_daily"] = df[rain_cols].max(axis=1)
df["runoff_event_7d"] = df[[f"runoff_day_{d}" for d in range(245, 252)]].sum(axis=1)
df["runoff_antecedent_30d"] = df[[f"runoff_day_{d}" for d in range(215, 245)]].sum(axis=1)
df["runoff_max_daily"] = df[runoff_cols].max(axis=1)

# UPDATED: Added "hand" to static_cols
static_cols = ["dem", "slope", "twi", "landcover", "soil_sand", "soil_clay", 
               "soil_cfvo", "snow_band_1", "snow_band_2", "rain_norm_band_1", "rain_norm_band_2", "hand"]
for col in static_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

df.rename(columns={"snow_band_1": "snow_winter", "snow_band_2": "snow_spring", 
                   "rain_norm_band_1": "rain_annual_norm", "rain_norm_band_2": "rain_monsoon_norm"}, inplace=True)

# UPDATED: Added "hand" to final_cols
final_cols = [
    "is_flood", "dem", "slope", "twi", "landcover",
    "soil_sand", "soil_clay", "soil_cfvo",
    "rain_annual_norm", "rain_monsoon_norm", "snow_winter", "snow_spring", "hand",
    "rain_event_7d", "rain_antecedent_30d", "rain_max_daily",
    "runoff_event_7d", "runoff_antecedent_30d", "runoff_max_daily"
]

df_final = df[final_cols].copy()
df_final.to_csv(OUT_PATH, index=False)
print(f"Saved clean matrix (18 features + target) to: {OUT_PATH}")