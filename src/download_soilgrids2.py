# src/download_soilgrids2.py — Stream SoilGrids via Remote VRTs (v5)
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

PROPS = ["sand", "clay", "cfvo"]
DEPTH = "5-15cm"
BBOX_4326 = (73.0, 32.0, 81.0, 38.0)  # left, bottom, right, top in degrees

# Exact URLs derived from the WebDAV directory listing
URLS = {
    "sand": "https://files.isric.org/soilgrids/latest/data/sand/sand_5-15cm_mean.vrt",
    "clay": "https://files.isric.org/soilgrids/latest/data/clay/clay_5-15cm_mean.vrt",
    "cfvo": "https://files.isric.org/soilgrids/latest/data/cfvo/cfvo_5-15cm_mean.vrt",
}

for p in PROPS:
    url = URLS[p]
    print(f"Connecting to remote VRT for {p}...")
    
    with rasterio.open(url) as src:
        print(f"  Native CRS: {src.crs}")
        
        # Translate degrees -> meters using pyproj
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        left, bottom = transformer.transform(BBOX_4326[0], BBOX_4326[1])
        right, top = transformer.transform(BBOX_4326[2], BBOX_4326[3])
        bbox_native = (left, bottom, right, top)
        
        # Calculate the exact window of pixels covering J&K
        win = from_bounds(*bbox_native, src.transform)
        win = win.round_offsets().round_lengths() # Ensure integer pixels
        print(f"  Windowed read: {win.width}x{win.height} pixels")
        
        # Download ONLY the data inside our window
        data = src.read(1, window=win)
        t = rasterio.windows.transform(win, src.transform)

        profile = src.profile.copy()
        profile.update(driver="GTiff", width=win.width, height=win.height,
                       transform=t, compress="deflate", tiled=False)
        
        out = rf"data\raw\soil\soilgrids_{p}_{DEPTH}.tif"
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data, 1)

        arr = data.astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
            
        print(f"  SAVED {out} | shape={data.shape}")
        print(f"  stats: min={np.nanmin(arr):.0f} max={np.nanmax(arr):.0f} nodata={src.nodata}")

print("DONE")