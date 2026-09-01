# src/build_ls_inventory.py — Landslide temporal inventory (Fixed exact columns)
import pandas as pd
import glob, os

cands = [p for p in glob.glob(r"data\raw\*") if 'landslide' in os.path.basename(p).lower() and p.endswith('.csv')]
print("Candidate catalog files:", cands)
if not cands:
    raise SystemExit("No landslide CSV in data\\raw")

df = pd.read_csv(cands[0], on_bad_lines='skip')
print(f"Loaded {len(df)} global records from {os.path.basename(cands[0])}")

# EXACT column matching (no substring traps)
lat_col  = next((c for c in df.columns if c.lower().strip() in ['latitude', 'lat']), None)
lon_col  = next((c for c in df.columns if c.lower().strip() in ['longitude', 'lon', 'lng']), None)
date_col = next((c for c in df.columns if c.lower().strip() in ['event_date', 'date']), None)

print(f"Detected: lat={lat_col} | lon={lon_col} | date={date_col}")
if not (lat_col and lon_col and date_col):
    raise SystemExit("Column detection failed.")

df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[lat_col, lon_col, 'Date'])

# Keep only hydro-meteorological triggers (a weather EWS must learn weather triggers;
# earthquake-triggered slides are label noise)
trig_col = next((c for c in df.columns if 'trigger' in c.lower()), None)
if trig_col:
    mask = df[trig_col].astype(str).str.lower().str.contains(
        'rain|snow|monsoon|storm|cloudburst|downpour|melt|flood', na=False)
    print(f"Trigger filter: {mask.sum()} of {len(df)} global events are hydro-meteorological")
    df = df[mask]

jk = df[(df[lat_col] >= 32) & (df[lat_col] <= 37.5) & (df[lon_col] >= 73) & (df[lon_col] <= 80.5)]
jk = jk[jk['Date'] >= '2000-01-01'].sort_values('Date')

print(f"\n=== J&K LANDSLIDE EVENTS (2000+): {len(jk)} ===")
print(jk[['Date', lat_col, lon_col]].head(10).to_string(index=False))

jk[[ 'Date', lat_col, lon_col]].rename(columns={lat_col: 'lat', lon_col: 'lon'}).to_csv(
    r"data\processed\j_k_landslide_inventory.csv", index=False)
print("\nSaved j_k_landslide_inventory.csv")