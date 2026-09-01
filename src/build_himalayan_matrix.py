# src/build_himalayan_matrix.py — Phase 8.2: Climatologically Normalized Matrix
import pandas as pd
import numpy as np
import os
import joblib

REGIONS = ['jk', 'himachal', 'uttarakhand', 'nepal']
SEASONS = {1: 'WD', 2: 'WD', 3: 'WD', 4: 'PRE', 5: 'PRE', 6: 'PRE', 
           7: 'MON', 8: 'MON', 9: 'MON', 10: 'POST', 11: 'POST', 12: 'POST'}

def load_series(path, target):
    d = pd.read_csv(path)
    d.columns = [str(c).strip() for c in d.columns]
    for junk in ['system:index', '.geo', 'system:time_start']:
        if junk in d.columns: d = d.drop(columns=[junk])
    date_col = next((c for c in d.columns if c.lower() == 'date'), None)
    if target not in d.columns:
        val_col = next((c for c in d.columns if c != date_col), None)
        if val_col: d = d.rename(columns={val_col: target})
    d['date'] = pd.to_datetime(d['date'])
    if target in d.columns:
        d[target] = pd.to_numeric(d[target], errors='coerce').replace(-9999, np.nan)
    return d[['date', target]] if target in d.columns else d[['date']]

all_dfs = []

for reg in REGIONS:
    print(f"\n--- Processing {reg.upper()} ---")
    chirps = load_series(rf"data\raw\CHIRPS_{reg}_TS.csv", 'rain_mean')
    chirps = chirps.sort_values('date').reset_index(drop=True)
    
    for w in [7, 30, 90]:
        chirps[f'rain_{w}d'] = chirps['rain_mean'].rolling(window=w).sum()
        
    chirps['month'] = chirps['date'].dt.month
    chirps['season'] = chirps['month'].map(SEASONS)
    
    # Build 42-year climatology lookup (Percentile ranks)
    climatology = {}
    for w in [7, 30, 90]:
        col = f'rain_{w}d'
        climatology[col] = {}
        for season in SEASONS.values():
            vals = chirps.loc[chirps['season'] == season, col].dropna()
            climatology[col][season] = np.sort(vals.values) if len(vals) > 0 else np.array([0])

    snow = load_series(rf"data\raw\SNOW_{reg}_TS.csv", 'snow_cover')
    lst = load_series(rf"data\raw\LST_{reg}_TS.csv", 'lst_day')
    
    df = chirps.merge(snow, on='date', how='left').merge(lst, on='date', how='left')
    df = df[df['date'] >= '2000-01-01'].copy() # Restrict to MODIS era
    
    df['snow_cover'] = df['snow_cover'].ffill(limit=3).bfill().fillna(0)
    df['lst_day'] = df['lst_day'].ffill(limit=3).bfill().fillna(df['lst_day'].median())
    
    # Map 2000-2023 sums to their seasonal percentiles
    for w in [7, 30, 90]:
        col = f'rain_{w}d'
        p_col = f'{col}_p'
        df[p_col] = np.nan
        for season in SEASONS.values():
            mask = df['season'] == season
            if mask.sum() > 0:
                arr = climatology[col][season]
                df.loc[mask, p_col] = np.searchsorted(arr, df.loc[mask, col].values) / len(arr)
                
    df['max_rain_3d'] = df['rain_mean'].rolling(3).max()
    df['snow_melt_7d'] = (df['snow_cover'].shift(7) - df['snow_cover']).clip(lower=0)
    df['melt_energy'] = df['lst_day'] * df['snow_melt_7d']
    df['snow_7d_max'] = df['snow_cover'].rolling(7).max()
    
    # Rain-on-snow flag
    df['ros_flag'] = ((df['rain_7d'] >= 20) & (df['snow_7d_max'] >= 10)).astype(int)
    
    df['rec_30_90'] = df['rain_30d'] / df['rain_90d'].clip(lower=1.0)
    df['rec_7_30'] = df['rain_7d'] / df['rain_30d'].clip(lower=1.0)
    
    doy = df['date'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    
    inv_path = rf"data\processed\flood_inventory_{reg}.csv"
    df['is_flood'] = 0
    if os.path.exists(inv_path):
        inv = pd.read_csv(inv_path)
        inv['Began'] = pd.to_datetime(inv['Began'])
        inv['Ended'] = pd.to_datetime(inv['Ended'])
        for _, ev in inv.iterrows():
            mask = (df['date'] >= ev['Began'] - pd.Timedelta(days=7)) & (df['date'] <= ev['Ended'])
            df.loc[mask, 'is_flood'] = 1
    else:
        print(f"Warning: Inventory for {reg} not found.")
        
    if reg == 'jk':
        joblib.dump(climatology, r"data\processed\climatology_jk.joblib")
        print("  Saved climatology_jk.joblib")
    df['region'] = reg
    df = df.dropna(subset=['rain_7d_p', 'rain_30d_p', 'rain_90d_p', 'snow_cover', 'lst_day']).copy()
    print(f"  Rows: {len(df)}, Flood days: {df['is_flood'].sum()}")
    all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True)
feature_cols = ['rain_7d_p', 'rain_30d_p', 'rain_90d_p', 'max_rain_3d',
                'snow_cover', 'snow_melt_7d', 'lst_day', 'melt_energy', 'ros_flag',
                'rec_30_90', 'rec_7_30', 'doy_sin', 'doy_cos']

final_df = final_df[['date', 'region'] + feature_cols + ['is_flood']]
print(f"\n=== FINAL HIMALAYAN MATRIX ===")
print(f"Shape: {final_df.shape}")
print(f"Total flood days: {final_df['is_flood'].sum()}")
print(final_df.groupby('region')['is_flood'].sum())
final_df.to_csv(r"data\processed\himalayan_flood_matrix.csv", index=False)
print("Saved himalayan_flood_matrix.csv")