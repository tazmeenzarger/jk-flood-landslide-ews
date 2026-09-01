# src/preflight_check.py — Deployment pre-flight verification
import os
import sys

def size_mb(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return None

print("=" * 60)
print("1. REQUIRED FILES CHECK")
print("=" * 60)

required = {
    "Flood raster":       "data/processed/evaluations_flood/flood_susceptibility_map.tif",
    "J&K boundary (.shp)":"data/processed/jk_boundary.shp",
    "Pooled model":       "data/processed/evaluations_temporal/temporal_flood_pooled.joblib",
    "Climatology":        "data/processed/climatology_jk.joblib",
}

ls_candidates = [
    "data/processed/evaluations/susceptibility_map.tif",
    "data/processed/evaluations/landslide_susceptibility_map.tif",
    "data/processed/evaluations/ls_susceptibility_map.tif",
]

all_ok = True
too_big = []

for label, path in required.items():
    if os.path.exists(path):
        mb = size_mb(path)
        flag = "OK" if mb < 100 else "TOO BIG for GitHub"
        if mb >= 100: too_big.append(path)
        print(f"  [{flag}] {label:<22} {mb:>8.2f} MB  ({path})")
    else:
        print(f"  [MISSING] {label:<22} {path}")
        all_ok = False

# Landslide map: at least one must exist
ls_found = next((p for p in ls_candidates if os.path.exists(p)), None)
if ls_found:
    mb = size_mb(ls_found)
    flag = "OK" if mb < 100 else "TOO BIG for GitHub"
    if mb >= 100: too_big.append(ls_found)
    print(f"  [{flag}] {'Landslide raster':<22} {mb:>8.2f} MB  ({ls_found})")
else:
    print(f"  [MISSING] Landslide raster (none of the candidates exist)")
    all_ok = False

# Shapefile companions (.shx, .dbf, .prj)
print("\n" + "=" * 60)
print("2. SHAPEFILE COMPANION FILES")
print("=" * 60)
base = "data/processed/jk_boundary"
for ext in [".shp", ".shx", ".dbf", ".prj"]:
    p = base + ext
    print(f"  [{'OK' if os.path.exists(p) else 'MISSING'}] {p}")
    if not os.path.exists(p):
        all_ok = False

print("\n" + "=" * 60)
print("3. LIBRARY IMPORTS")
print("=" * 60)
for lib in ["streamlit", "rasterio", "geopandas", "xgboost", "joblib", "matplotlib"]:
    try:
        __import__(lib)
        print(f"  [OK] {lib}")
    except ImportError:
        print(f"  [FAIL] {lib}")
        all_ok = False

print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
if too_big:
    print(f"  WARNING: {len(too_big)} file(s) exceed GitHub's 100MB limit.")
    for p in too_big: print(f"    - {p}")
print("  GO for GitHub upload" if all_ok else "  NO-GO: fix the MISSING/FAIL items first")