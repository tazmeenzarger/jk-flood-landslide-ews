# src/engineer_features.py — Phase 3.2: Hydrological Feature Engineering
import pandas as pd
import numpy as np

print("Loading raw features matrix...")
df = pd.read_csv(r"data\processed\features.csv")

# 1. INTELLIGENT NaN HANDLING
# Rain/Runoff: NaN means no measurement, safely assume 0
rain_cols = [c for c in df.columns if c.startswith("rain_2014")]
runoff_cols = [c for c in df.columns if c.startswith("runoff")]
df[rain_cols + runoff_cols] = df[rain_cols + runoff_cols].fillna(0)

# Static/Soil/Snow: Fill missing values (lakes/glaciers) with the median (robust to outliers)
static_cols = ["dem", "slope", "twi", "landcover", "soil_sand", "soil_clay", 
               "soil_cfvo", "snow_band_1", "snow_band_2", "rain_norm_band_1", "rain_norm_band_2"]
for col in static_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# 2. FEATURE ENGINEERING (The Shah et al. Method)
# Event window: Sept 2-8, 2014 (Days 245-251 in a non-leap year)
event_days_rain = [f"rain_2014_day_{d}" for d in range(245, 252)]
event_days_run = [f"runoff_day_{d}" for d in range(245, 252)]

df["rain_event_7d"] = df[event_days_rain].sum(axis=1)
df["runoff_event_7d"] = df[event_days_run].sum(axis=1)

# Antecedent window: 30 days prior to the event (Days 215-244)
ante_days_rain = [f"rain_2014_day_{d}" for d in range(215, 245)]
ante_days_run = [f"runoff_day_{d}" for d in range(215, 245)]

df["rain_antecedent_30d"] = df[ante_days_rain].sum(axis=1)
df["runoff_antecedent_30d"] = df[ante_days_run].sum(axis=1)

# Extreme triggers: The single heaviest day of the year
df["rain_max_daily"] = df[rain_cols].max(axis=1)
df["runoff_max_daily"] = df[runoff_cols].max(axis=1)

# Rename Snow bands for clarity
df.rename(columns={"snow_band_1": "snow_winter", "snow_band_2": "snow_spring"}, inplace=True)
df.rename(columns={"rain_norm_band_1": "rain_annual_norm", "rain_norm_band_2": "rain_monsoon_norm"}, inplace=True)

# 3. BUILD FINAL MATRIX
# We drop the 730 daily columns and keep only the engineered physics
final_cols = [
    "is_landslide", 
    "dem", "slope", "twi", "landcover",
    "soil_sand", "soil_clay", "soil_cfvo",
    "rain_annual_norm", "rain_monsoon_norm",
    "snow_winter", "snow_spring",
    "rain_event_7d", "rain_antecedent_30d", "rain_max_daily",
    "runoff_event_7d", "runoff_antecedent_30d", "runoff_max_daily"
]

df_final = df[final_cols].copy()

print(f"\n=== FEATURE ENGINEERING COMPLETE ===")
print(f"Original shape: {df.shape}")
print(f"Final shape:    {df_final.shape}")
print(f"Total NaNs remaining: {df_final.isna().sum().sum()}")

print("\nNew Engineered Features (Mean for Background vs Landslide):")
eng_cols = ["rain_event_7d", "rain_antecedent_30d", "runoff_event_7d", "runoff_antecedent_30d"]
print(df_final.groupby("is_landslide")[eng_cols].mean().round(2).to_string())

df_final.to_csv(r"data\processed\features_final.csv", index=False)
print(f"\nSaved clean matrix to: data\\processed\\features_final.csv")