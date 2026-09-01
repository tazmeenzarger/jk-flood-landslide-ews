# src/build_temporal_matrix.py — Phase 4.3: Temporal EWS Feature Matrix
import pandas as pd
import numpy as np

print("Loading daily time-series and flood inventory...")
# Load the CHIRPS daily rainfall data
ts = pd.read_csv(r"data\raw\CHIRPS_JK_Daily_TimeSeries_2000_2023.csv")
ts['date'] = pd.to_datetime(ts['date'])
ts = ts.sort_values('date').reset_index(drop=True)

# Load the multi-year flood inventory
inv = pd.read_csv(r"data\processed\j_k_flood_inventory.csv")
inv['Began'] = pd.to_datetime(inv['Began'])
inv['Ended'] = pd.to_datetime(inv['Ended'])

print(f"Time-series length: {len(ts)} days")
print(f"Inventory events: {len(inv)}")

# 1. ENGINEER FEATURES (Rolling Windows)
# These represent the hydrological "memory" of the basin
print("Engineering rolling hydrological features...")
ts['rain_1d'] = ts['rain_mean']
ts['rain_3d'] = ts['rain_mean'].rolling(window=3).sum()
ts['rain_7d'] = ts['rain_mean'].rolling(window=7).sum()
ts['rain_14d'] = ts['rain_mean'].rolling(window=14).sum()
ts['rain_30d'] = ts['rain_mean'].rolling(window=30).sum() # Antecedent saturation!
ts['max_rain_3d'] = ts['rain_mean'].rolling(window=3).max() # Extreme trigger

# 2. ENGINEER THE EWS LABEL (Forward-Looking)
# We label a day as '1' if a flood is CURRENTLY active OR WILL START within 7 days.
# This gives the model a 7-day Early Warning lead time!
print("Generating 7-day forward-looking Early Warning labels...")
ts['is_flood'] = 0
lead_time_days = 7

for index, event in inv.iterrows():
    start = event['Began']
    end = event['Ended']
    
    # Active flood days
    mask_active = (ts['date'] >= start) & (ts['date'] <= end)
    ts.loc[mask_active, 'is_flood'] = 1
    
    # Early warning days (7 days BEFORE the flood starts)
    mask_ews = (ts['date'] >= start - pd.Timedelta(days=lead_time_days)) & (ts['date'] < start)
    ts.loc[mask_ews, 'is_flood'] = 1

# 3. CLEANUP
# Drop the first 30 rows because the 30-day rolling sum needs 30 days of history to calculate
ts_clean = ts.dropna().copy()

# Keep only the features and the label for the ML model
feature_cols = ['rain_1d', 'rain_3d', 'rain_7d', 'rain_14d', 'rain_30d', 'max_rain_3d']
final_cols = ['date'] + feature_cols + ['is_flood']
df_final = ts_clean[final_cols].copy()

print(f"\n=== TEMPORAL EWS MATRIX COMPLETE ===")
print(f"Final matrix shape: {df_final.shape} (Rows = Days, Cols = Features + Target)")
print(f"Total days labeled as High Alert (1): {df_final['is_flood'].sum()}")
print(f"Total days labeled as Safe (0): {(df_final['is_flood'] == 0).sum()}")

print("\nSample of matrix during Sept 2014 Kashmir Floods:")
sept_2014 = df_final[(df_final['date'] >= '2014-08-20') & (df_final['date'] <= '2014-09-10')]
print(sept_2014[['date', 'rain_7d', 'rain_30d', 'is_flood']].to_string(index=False))

out_path = r"data\processed\temporal_ews_matrix.csv"
df_final.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")