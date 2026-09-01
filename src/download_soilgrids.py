# src/download_soilgrids.py
# Directly queries ISRIC's Web Coverage Service (WCS) for SoilGrids v2 data
import requests
import os

print("Connecting to ISRIC SoilGrids servers...")

# J&K Bounding Box (slightly padded for safety)
# x = longitude, y = latitude
bbox = {
    'x_min': 73.0, 'x_max': 81.0,
    'y_min': 32.0, 'y_max': 38.0
}

# Official ISRIC WCS endpoint
base_url = "https://maps.isric.org/mapserv"
params_template = {
    "map": "/datalocations/soilgrids.map",
    "SERVICE": "WCS",
    "VERSION": "2.0.1",
    "REQUEST": "GetCoverage",
    "FORMAT": "image/tiff",
    "SUBSETTINGCRS": "http://www.opengis.net/def/crs/OGC/0/CRS84",
    "SUBSET": [
        f"x({bbox['x_min']},{bbox['x_max']})",
        f"y({bbox['y_min']},{bbox['y_max']})"
    ]
}

# The actual SoilGrids variables we need
properties = ["sand", "clay", "cfvo"] # cfvo = coarse fragments (gravel/rocks)
depth = "5-15cm"

# Ensure the output folder exists
os.makedirs("data/raw/soil", exist_ok=True)

for prop in properties:
    coverage_id = f"{prop}_{depth}_mean"
    print(f"Downloading {coverage_id} (250m resolution)...")

    params = params_template.copy()
    params["COVERAGEID"] = coverage_id

    # Stream the download so it doesn't crash your RAM
    response = requests.get(base_url, params=params, stream=True)

    if response.status_code == 200:
        filepath = f"data/raw/soil/soilgrids_{prop}_{depth}.tif"
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  -> Success! Saved to: {filepath}")
    else:
        print(f"  -> Failed with status code: {response.status_code}")

print("\nSoilGrids download complete!")