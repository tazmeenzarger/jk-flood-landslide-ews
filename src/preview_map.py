# src/preview_map.py — Turn the susceptibility numbers into a colored picture
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

# 1. Read the model's probability numbers
with rasterio.open(r"data\processed\evaluations\susceptibility_map.tif") as src:
    data = src.read(1)                      # the 0-to-1 grid
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    crs = src.crs

# 2. Read the J&K border and match it to the same coordinate system
boundary = gpd.read_file(r"data\processed\jk_boundary.shp").to_crs(crs)

# 3. Draw it
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(data, extent=extent, cmap='Reds', vmin=0, vmax=1)
boundary.plot(facecolor='none', edgecolor='black', linewidth=1.5, ax=ax)
plt.colorbar(im, ax=ax, label='Landslide Probability (0 = safe, 1 = danger)')
ax.set_title('J&K Landslide Susceptibility (Random Forest)')
plt.tight_layout()

# 4. Save as a normal picture you can open
out = r"data\processed\evaluations\susceptibility_preview.png"
plt.savefig(out, dpi=150)
print(f"Saved picture to: {out}")