# src/check_runoff.py — prove the runoff file captured the 2014 flood
import numpy as np
import rasterio

path = r"data\raw\runoff\era5_runoff_2014.tif"
with rasterio.open(path) as src:
    names = list(src.descriptions)
    rows = []
    for i in range(1, src.count + 1):
        d = src.read(i).astype("float32")
        d = np.where(d < -1e30, np.nan, d)
        rows.append((names[i - 1], float(np.nanmax(d))))

rows.sort(key=lambda r: -r[1])
print("Top 10 runoff days of 2014 (band, max runoff in meters/day):")
for n, m in rows[:10]:
    print(f"  {n}  {m:.4f}")