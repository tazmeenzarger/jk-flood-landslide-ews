# src/eda_features.py — Phase 3.1: first look at the matrix
import pandas as pd

df = pd.read_csv(r"data\processed\features.csv")
print("shape:", df.shape)
print("\nclass balance:\n", df["is_landslide"].value_counts())

nan = df.isna().sum()
print("\ncolumns containing NaN:")
print(nan[nan > 0].to_string() if (nan > 0).any() else "  (none)")

static = ["dem", "slope", "twi", "rain_norm_band_1", "rain_norm_band_2",
          "landcover", "soil_sand", "soil_clay", "soil_cfvo",
          "snow_band_1", "snow_band_2"]
print("\nstatic feature stats:")
print(df[static].describe().round(2).to_string())

# Peek at the flood signal: mean event-week rain for landslides vs background
event_days = [c for c in df.columns if c.startswith("rain_2014_day_")]
df["_sept_rain"] = df[event_days[244:251]].sum(axis=1)   # Sept 2-8, 2014
print("\nmean Sept 2-8 rain (mm):")
print(df.groupby("is_landslide")["_sept_rain"].mean().round(2))