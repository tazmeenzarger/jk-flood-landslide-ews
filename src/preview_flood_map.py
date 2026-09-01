# src/preview_flood_map.py
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

# 1. Read the flood model's probability numbers
with rasterio.open(r"data\processed\evaluations_flood\flood_susceptibility_map.tif") as src:
    data = src.read(1)
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    crs = src.crs

# 2. Read the J&K border
boundary = gpd.read_file(r"data\processed\jk_boundary.shp").to_crs(crs)

# 3. Draw it (Using BLUES for floods)
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(data, extent=extent, cmap='Blues', vmin=0, vmax=1)
boundary.plot(facecolor='none', edgecolor='black', linewidth=1.5, ax=ax)
plt.colorbar(im, ax=ax, label='Flood Probability (0 = safe, 1 = danger)')
ax.set_title('J&K Flood Susceptibility (Random Forest)')
plt.tight_layout()

# 4. Save
out = r"data\processed\evaluations_flood\flood_preview.png"
plt.savefig(out, dpi=150)
print(f"Saved picture to: {out}")