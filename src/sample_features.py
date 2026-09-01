# src/sample_features.py — Phase 2.6: The Raster-to-Tabular Bridge (Optimized)
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

LABELS_PATH = r"data\processed\labels.gpkg"
OUT_PATH = r"data\processed\features.csv"

RASTERS = {
    "dem": r"data\processed\aligned\dem.tif",
    "slope": r"data\processed\aligned\slope.tif",
    "twi": r"data\processed\aligned\twi.tif",
    "rain_norm": r"data\processed\aligned\rain_norm.tif",
    "rain_2014": r"data\processed\aligned\rain_2014.tif",
    "landcover": r"data\processed\aligned\landcover.tif",
    "soil_sand": r"data\processed\aligned\soil_sand.tif",
    "soil_clay": r"data\processed\aligned\soil_clay.tif",
    "soil_cfvo": r"data\processed\aligned\soil_cfvo.tif",
    "snow": r"data\processed\aligned\snow.tif",
    "runoff": r"data\processed\aligned\runoff.tif",
}

print("Loading 1048 points...")
gdf = gpd.read_file(LABELS_PATH)
coords = list(zip(gdf.geometry.x, gdf.geometry.y))

# Start a list of DataFrames to concatenate at the end
dfs = [pd.DataFrame({'is_landslide': gdf['is_landslide']})]

print("Interrogating rasters (Sampling)...")
for name, path in RASTERS.items():
    # end=" " and flush=True keep the cursor on the same line so you see progress
    print(f"  Sampling {name}...", end=" ", flush=True)
    
    with rasterio.open(path) as src:
        # Stack the sampled values into a 2D numpy array
        vals = np.vstack(list(src.sample(coords))).astype("float32")
        
        # Replace NoData with NaN safely
        if src.nodata is not None:
            vals = np.where(vals == src.nodata, np.nan, vals)
            
        if src.count == 1:
            dfs.append(pd.DataFrame({name: vals[:, 0]}))
        else:
            # Generate column names dynamically
            cols = [f"{name}_day_{i+1}" if name in ['rain_2014', 'runoff'] else f"{name}_band_{i+1}" for i in range(src.count)]
            dfs.append(pd.DataFrame(vals, columns=cols))
            
    print("Done.")

print("\nMerging and saving to CSV...")
# This is the professional way: glue them all together at once
df = pd.concat(dfs, axis=1)
df.to_csv(OUT_PATH, index=False)

print(f"\n=== RASTER-TO-TABULAR BRIDGE COMPLETE ===")
print(f"Saved to: {OUT_PATH}")
print(f"Matrix shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Class balance: {df['is_landslide'].sum()} landslides vs {(df['is_landslide']==0).sum()} background")
print("\nPHASE 2 IS OFFICIALLY COMPLETE! 🎉")