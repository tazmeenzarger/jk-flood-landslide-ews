# src/build_ls_temporal_matrix.py — Landslide temporal matrix (final)
import pandas as pd
import numpy as np

def load_series(path, target):
    d = pd.read_csv(path)
    d.columns = [str(c).strip() for c in d.columns]
    for junk in ['system:index', '.geo', 'system:time_start']:
        if junk in d.columns:
            d = d.drop(columns=[junk])
    date_col = next((c for c in d.columns if c.lower() == 'date'), None)
    if target not in d.columns:
        val_col = next((c for c in d.columns if c != date_col), None)
        d = d.rename(columns={val_col: target})
    d['date'] = pd.to_datetime(d['date'])
    d[target] = pd.to_numeric(d[target], errors='coerce').replace(-9999, np.nan)
    return d[['date', target]]

ts = load_series(r"data\raw\CHIRPS_JK_Daily_TimeSeries_2000_2023.csv", 'rain_mean')
snow = load_series(r"data\raw\MODIS_SnowCover_TS_V2.csv", 'snow_cover')
lst = load_series(r"data\raw\MODIS_LST_TS_V2.csv", 'lst_day')

ts = ts.merge(snow, on='date', how='left').merge(lst, on='date', how='left')
ts = ts.sort_values('date').reset_index(drop=True)
ts['snow_cover'] = ts['snow_cover'].ffill(limit=3).bfill().fillna(0)
ts['lst_day'] = ts['lst_day'].ffill(limit=3).bfill().fillna(ts['lst_day'].median())

for w in [1, 3, 7, 14, 30, 60, 90]:
    ts[f'rain_{w}d'] = ts['rain_mean'].rolling(window=w).sum()
ts['max_rain_3d'] = ts['rain_mean'].rolling(3).max()
ts['snow_melt_7d'] = (ts['snow_cover'].shift(7) - ts['snow_cover']).clip(lower=0)
ts['melt_energy'] = ts['lst_day'] * ts['snow_melt_7d']
doy = ts['date'].dt.dayofyear
ts['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
ts['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
ts['rec_30_90'] = ts['rain_30d'] / ts['rain_90d'].clip(lower=1.0)
ts['rec_7_30'] = ts['rain_7d'] / ts['rain_30d'].clip(lower=1.0)

# LANDSLIDE LABELS: event day + 3-day lead window (fast-response hazard)
inv = pd.read_csv(r"data\processed\j_k_landslide_inventory.csv")
inv['Date'] = pd.to_datetime(inv['Date'])
ts['is_ls'] = 0
for _, ev in inv.iterrows():
    ts.loc[(ts['date'] >= ev['Date'] - pd.Timedelta(days=3)) & (ts['date'] <= ev['Date']), 'is_ls'] = 1

feature_cols = ['rain_1d', 'rain_3d', 'rain_7d', 'rain_14d', 'rain_30d', 'rain_60d', 'rain_90d',
                'max_rain_3d', 'snow_cover', 'snow_melt_7d', 'lst_day', 'melt_energy', 'doy_sin', 'doy_cos',
                'rec_30_90', 'rec_7_30']

df_final = ts.dropna(subset=feature_cols).copy()
df_final = df_final[['date'] + feature_cols + ['is_ls']]
print(f"Final matrix: {df_final.shape} | Alert days: {df_final['is_ls'].sum()}")
df_final.to_csv(r"data\processed\temporal_ls_matrix.csv", index=False)
print("Saved temporal_ls_matrix.csv")