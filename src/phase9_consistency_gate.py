# src/phase9_consistency_gate.py — Phase 9.1: Local CSV consistency gate
import pandas as pd
import numpy as np

print("Loading local Open-Meteo CSV...")
om = pd.read_csv(r"data\raw\openmeteo_srinagar.csv", skiprows=2)
# Robust column detection (Open-Meteo sometimes names them 'time' and 'precipitation_sum')
date_col = next(c for c in om.columns if 'time' in c.lower() or 'date' in c.lower())
rain_col = next(c for c in om.columns if 'precip' in c.lower() or 'rain' in c.lower())
om = om.rename(columns={date_col: "date", rain_col: "om_rain"})
om["date"] = pd.to_datetime(om["date"])

ch = pd.read_csv(r"data\raw\CHIRPS_JK_Daily_TimeSeries_2000_2023.csv")
ch.columns = [str(c).strip() for c in ch.columns]
ch = ch.rename(columns={"rain_mean": "ch_rain"})
ch["date"] = pd.to_datetime(ch["date"])

m = om.merge(ch[["date", "ch_rain"]], on="date", how="inner").dropna()
print(f"\nOverlap days: {len(m)}")

for w in [1, 7, 30]:
    m[f"om_{w}d"] = m["om_rain"].rolling(w).sum()
    m[f"ch_{w}d"] = m["ch_rain"].rolling(w).sum()

mm = m.dropna(subset=["om_30d", "ch_30d"])
print("\n=== PEARSON r (Open-Meteo ERA5 vs CHIRPS basin mean) ===")
for w in [1, 7, 30]:
    print(f"{w:>2}-day sums: r = {mm[f'ch_{w}d'].corr(mm[f'om_{w}d']):.3f}")

print(f"\n30d mean bias ratio (OM/CHIRPS): {mm['om_30d'].mean() / mm['ch_30d'].mean():.2f}")

SEASONS = {1:'WD',2:'WD',3:'WD',4:'PRE',5:'PRE',6:'PRE',7:'MON',8:'MON',9:'MON',10:'POST',11:'POST',12:'POST'}
mm["season"] = mm["date"].dt.month.map(SEASONS)
print("\n30d r by season:")
for s in ["WD", "PRE", "MON", "POST"]:
    sub = mm[mm["season"] == s]
    print(f"  {s:>4}: r = {sub['ch_30d'].corr(sub['om_30d']):.3f} (n={len(sub)})")

r30 = mm["ch_30d"].corr(mm["om_30d"])
print("\n=== GATE VERDICT ===")
print("GO - build Live Monitor" if r30 >= 0.7 else "NO-GO - document and skip live mode")