# Jammu & Kashmir Multi-Hazard Early Warning System

An operational prototype that converts raw satellite data into actionable flood and 
landslide warnings for the Jammu & Kashmir Himalayan basin.

## Overview
The system couples **spatial susceptibility** (where hazards can occur) with 
**temporal early warning** (when they are likely), delivered through an interactive 
Streamlit dashboard.

| Component | Model | Performance |
|---|---|---|
| Flood susceptibility | Random Forest | AUC 0.984 |
| Landslide susceptibility | Random Forest | AUC 0.938 |
| Flood temporal EWS | Himalayan-pooled XGBoost | AUC 0.750, ~6-day lead |
| Landslide trigger | Rule-based advisory | (catalog gap post-2018) |

## Key Methodology
- **42-year climatology (1981–2023)** normalization of CHIRPS rainfall into seasonal percentiles.
- **Himalayan regional pooling** across J&K, Himachal, Uttarakhand, and Nepal to compensate 
  for local catalog sparsity.
- **Data sources:** CHIRPS, MODIS (snow/LST), SRTM, HAND hydrology, DFO & NASA hazard catalogs.

## Running the Dashboard
pip install -r requirements.txt
streamlit run src/dashboard.py


## Repository Structure
- `src/dashboard.py` — the deployed Streamlit app
- `src/*.py` — training and data-engineering pipeline
- `data/processed/` — trained models, susceptibility maps, and boundary files

## Author
Tazmeen Zargar — Personal project, 2026. All rights reserved (see LICENSE).