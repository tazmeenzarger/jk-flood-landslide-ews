# src/predict_flood_susceptibility.py — Path A Update
import os
import numpy as np
import rasterio
import joblib

ALIGNED_DIR = r"data\processed\aligned"
EVAL_DIR = r"data\processed\evaluations_flood"

print("Loading flood model and scaler...")
model = joblib.load(os.path.join(EVAL_DIR, "flood_Random_Forest.joblib"))
scaler = joblib.load(os.path.join(EVAL_DIR, "flood_scaler.joblib"))

def read_band(path, band_idx=1):
    with rasterio.open(path) as src:
        data = src.read(band_idx).astype("float32")
        if src.nodata is not None: data[data == src.nodata] = np.nan
        return data

print("Loading base rasters...")
dem = read_band(os.path.join(ALIGNED_DIR, "dem.tif"))
slope = read_band(os.path.join(ALIGNED_DIR, "slope.tif"))
twi = read_band(os.path.join(ALIGNED_DIR, "twi.tif"))
hand = read_band(os.path.join(ALIGNED_DIR, "hand.tif")) # ADDED HAND
landcover = read_band(os.path.join(ALIGNED_DIR, "landcover.tif"))
soil_sand = read_band(os.path.join(ALIGNED_DIR, "soil_sand.tif"))
soil_clay = read_band(os.path.join(ALIGNED_DIR, "soil_clay.tif"))
soil_cfvo = read_band(os.path.join(ALIGNED_DIR, "soil_cfvo.tif"))

with rasterio.open(os.path.join(ALIGNED_DIR, "snow.tif")) as src:
    snow_winter = src.read(1).astype("float32"); snow_spring = src.read(2).astype("float32")
    snow_winter[snow_winter == src.nodata] = np.nan; snow_spring[snow_spring == src.nodata] = np.nan
with rasterio.open(os.path.join(ALIGNED_DIR, "rain_norm.tif")) as src:
    rain_annual = src.read(1).astype("float32"); rain_monsoon = src.read(2).astype("float32")
    rain_annual[rain_annual == src.nodata] = np.nan; rain_monsoon[rain_monsoon == src.nodata] = np.nan

print("Engineering daily rain and runoff grids...")
with rasterio.open(os.path.join(ALIGNED_DIR, "rain_2014.tif")) as src:
    rain_365 = src.read() 
    if src.nodata is not None: rain_365[rain_365 == src.nodata] = np.nan
    rain_365 = np.nan_to_num(rain_365, nan=0.0)
    rain_event_7d = rain_365[244:251].sum(axis=0)
    rain_antecedent_30d = rain_365[214:244].sum(axis=0)
    rain_max_daily = rain_365.max(axis=0)
with rasterio.open(os.path.join(ALIGNED_DIR, "runoff.tif")) as src:
    runoff_365 = src.read()
    if src.nodata is not None: runoff_365[runoff_365 == src.nodata] = np.nan
    runoff_365 = np.nan_to_num(runoff_365, nan=0.0)
    runoff_event_7d = runoff_365[244:251].sum(axis=0)
    runoff_antecedent_30d = runoff_365[214:244].sum(axis=0)
    runoff_max_daily = runoff_365.max(axis=0)

print("Stacking features...")
# UPDATED STACK ORDER to match the training CSV exactly
features_3d = np.stack([
    dem, slope, twi, landcover,
    soil_sand, soil_clay, soil_cfvo,
    rain_annual, rain_monsoon,
    snow_winter, snow_spring,
    hand, # ADDED HAND AT INDEX 11
    rain_event_7d, rain_antecedent_30d, rain_max_daily,
    runoff_event_7d, runoff_antecedent_30d, runoff_max_daily
], axis=-1) 

height, width, n_features = features_3d.shape
pixels = features_3d.reshape(-1, n_features)

print("Imputing NaNs...")
# UPDATED RANGES: Engineered features are now at indices 12-17
for i in range(12, 18): 
    pixels[:, i] = np.nan_to_num(pixels[:, i], nan=0.0)

# Static features are now at indices 0-11
for i in range(12): 
    valid_vals = pixels[:, i][~np.isnan(pixels[:, i])]
    if len(valid_vals) > 0:
        med = np.median(valid_vals)
        pixels[:, i] = np.nan_to_num(pixels[:, i], nan=med)

print("Scaling...")
pixels_scaled = scaler.transform(pixels)
probs = model.predict_proba(pixels_scaled)[:, 1]
probs_2d = probs.reshape(height, width)

with rasterio.open(os.path.join(ALIGNED_DIR, "dem.tif")) as src:
    transform = src.transform; crs = src.crs

out_path = os.path.join(EVAL_DIR, "flood_susceptibility_map.tif")
with rasterio.open(out_path, 'w', driver='GTiff', height=height, width=width, count=1,
    dtype='float32', crs=crs, transform=transform, nodata=-9999.0, compress='lzw') as dst:
    dst.write(probs_2d, 1)
print("Flood Susceptibility Map Generation Complete! 🎉")