# src/extract_jk.py
# The "cookie cutter": slices J&K out of the India map
import glob
import geopandas as gpd

# 1. Find the Level 1 shapefile wherever it sits inside data/raw
shp_path = glob.glob("data/raw/**/gadm41_IND_1.shp", recursive=True)[0]
india = gpd.read_file(shp_path)

# 2. Print every state/UT name so we see exactly what GADM calls them
print(india["NAME_1"].to_list())

# 3. THE COOKIE CUTTER: keep only the polygon(s) whose name contains "Jammu"
jk = india[india["NAME_1"].str.contains("Jammu", case=False, na=False)]
print(f"Found {len(jk)} polygon(s) for J&K")

# 4. Save the sliced J&K boundary as its own clean file
jk.to_file("data/processed/jk_boundary.shp")
print("Saved to data/processed/jk_boundary.shp")