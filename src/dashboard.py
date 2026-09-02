# src/dashboard.py — J&K Multi-Hazard EWS (final consolidated edition)
import streamlit as st
import joblib
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime
import os

def precise_slider(label, min_v, max_v, default, step, key):
    """Slider + synced exact number input."""
    if key + "_s" not in st.session_state:
        st.session_state[key + "_s"] = default
        st.session_state[key + "_n"] = default
    c1, c2 = st.sidebar.columns([3, 2])
    with c1:
        st.slider(label, min_v, max_v, step=step, key=key + "_s",
                  on_change=lambda: st.session_state.update({key + "_n": st.session_state[key + "_s"]}))
    with c2:
        st.number_input("exact", min_v, max_v, step=step, key=key + "_n", label_visibility="collapsed",
                        on_change=lambda: st.session_state.update({key + "_s": st.session_state[key + "_n"]}))
    return st.session_state[key + "_s"]

st.set_page_config(page_title="J&K Multi-Hazard EWS", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .status-card { padding: 18px 22px; border-radius: 6px; margin: 8px 0 16px 0; }
    .status-card h3 { margin: 0 0 6px 0; font-size: 1.3rem; letter-spacing: 0.4px; }
    .status-card p  { margin: 0; font-size: 0.95rem; line-height: 1.55; }
    .status-safe  { background-color: #e8f5e9; border-left: 5px solid #2e7d32; }
    .status-safe  h3 { color: #1b5e20; }
    .status-safe  p  { color: #33502f; }
    .status-watch { background-color: #fff3e0; border-left: 5px solid #ef6c00; }
    .status-watch h3 { color: #e65100; }
    .status-watch p  { color: #5d4037; }
    .status-alert { background-color: #fdecea; border-left: 5px solid #b71c1c; }
    .status-alert h3 { color: #b71c1c; }
    .status-alert p  { color: #5f2120; }
    .info-box { background-color: #f0f4f8; border-radius: 6px; padding: 15px; margin-top: 20px; border: 1px solid #d1d9e0; }
</style>
""", unsafe_allow_html=True)

st.title("Jammu & Kashmir Multi-Hazard Early Warning System")
st.caption("Operational Prototype | Spatial Susceptibility & Temporal Early Warning")
st.markdown("---")

# --- PATHS ---
FLOOD_MAP = r"data/processed/evaluations_flood/flood_susceptibility_map.tif"
LS_CANDIDATES = [r"data/processed/evaluations/susceptibility_map.tif",
                 r"data/processed/evaluations/landslide_susceptibility_map.tif",
                 r"data/processed/evaluations/ls_susceptibility_map.tif"]
BOUNDARY = r"data/processed/jk_boundary.shp"
TEMPORAL_MODEL_PATH = r"data/processed/evaluations_temporal/temporal_flood_pooled.joblib"
CLIM_PATH = r"data/processed/climatology_jk.joblib"
THRESHOLD = 0.34

# --- LOADERS ---
@st.cache_data
def load_raster(path):
    with rasterio.open(path) as src:
        return src.read(1), src.bounds, src.crs

@st.cache_data
def load_boundary():
    return gpd.read_file(BOUNDARY)

boundary = load_boundary()
flood_map, flood_bounds, raster_crs = load_raster(FLOOD_MAP)
ls_path = next((p for p in LS_CANDIDATES if os.path.exists(p)), None)
if ls_path:
    ls_map, ls_bounds, _ = load_raster(ls_path)
else:
    ls_map, ls_bounds = None, None

temporal_model = joblib.load(TEMPORAL_MODEL_PATH)
_clim = joblib.load(CLIM_PATH)

# --- SHARED MAP HELPER ---
def draw_map(ax, base, bounds, crs, dynamic, cmap, mask_below):
    ext = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    ax.imshow(base, cmap="gray", alpha=0.18, extent=ext)
    risk = np.ma.masked_where(dynamic <= mask_below, dynamic)
    im = ax.imshow(risk, cmap=cmap, vmin=0, vmax=1, extent=ext)
    boundary.to_crs(crs.to_wkt()).plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.4)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im

blues = mcolors.LinearSegmentedColormap.from_list("custom_blues",
        ["#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"], N=256)
reds = mcolors.LinearSegmentedColormap.from_list("custom_reds",
        ["#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"], N=256)

# --- SIDEBAR (precise controls) ---
st.sidebar.title("Forecast Parameters")
guard_slot = st.sidebar.empty()
st.sidebar.markdown("#### Precipitation Dynamics")
r7   = precise_slider("7-Day Rain (mm)", 0.0, 500.0, 30.0, 5.0, "r7")
r30  = precise_slider("30-Day Rain (mm)", 0.0, 1500.0, 100.0, 5.0, "r30")
r90  = precise_slider("90-Day Rain (mm)", 0.0, 5000.0, 400.0, 10.0, "r90")
max3 = precise_slider("Max 3-Day (mm)", 0.0, 200.0, 10.0, 5.0, "max3")

st.sidebar.markdown("#### Cryosphere & Environment")
snow = precise_slider("Snow Cover (%)", 0.0, 100.0, 20.0, 1.0, "snow")
melt = precise_slider("7-Day Snow Melt (%)", 0.0, 100.0, 5.0, 1.0, "melt")
lst  = precise_slider("Surface Temp (K)", 250.0, 320.0, 290.0, 1.0, "lst")

# --- TEMPORAL MODEL INPUTS (climatologically normalized) ---
melt_energy = lst * melt
doy = datetime.date.today().timetuple().tm_yday
doy_sin = np.sin(2 * np.pi * doy / 365.25)
doy_cos = np.cos(2 * np.pi * doy / 365.25)

SEASON_OF_MONTH = {1:'WD',2:'WD',3:'WD',4:'PRE',5:'PRE',6:'PRE',
                   7:'MON',8:'MON',9:'MON',10:'POST',11:'POST',12:'POST'}
_cur_season = SEASON_OF_MONTH[datetime.date.today().month]

def pct(window, value):
    arr = _clim[f'rain_{window}d'][_cur_season]
    return float(np.searchsorted(arr, value) / len(arr))

rain_7d_p, rain_30d_p, rain_90d_p = pct(7, r7), pct(30, r30), pct(90, r90)
rec_30_90 = r30 / max(r90, 1.0)
rec_7_30 = r7 / max(r30, 1.0)
ros_flag = int((r7 >= 20) and (snow >= 10))
# --- INPUT CONSISTENCY GUARDS ---
guard_msgs = []
if r7 > r30 or r30 > r90:
    guard_msgs.append("Rain windows must nest (7d <= 30d <= 90d).")
if max3 > r30 or max3 > r7:
    guard_msgs.append("Max 3-day rain cannot exceed the 7-day and 30-day totals.")
if (rec_30_90 > 0.65) and (rain_30d_p > 0.90):
    guard_msgs.append("30-day total is an extreme share of the 90-day total - outside the observed climatology.")
if guard_msgs:
    guard_slot.warning("Manifold guard: " + " ".join(guard_msgs))

features = np.array([[rain_7d_p, rain_30d_p, rain_90d_p, max3,
                      snow, melt, lst, melt_energy, ros_flag,
                      rec_30_90, rec_7_30, doy_sin, doy_cos]])
temporal_prob = float(temporal_model.predict_proba(features)[0][1])
burst = (max3 >= 30) or (r7 >= 40)

# --- TABS ---
tab_flood, tab_ls = st.tabs(["Flood Early Warning", "Landslide Hazard"])

# ================= TAB 1: FLOOD EARLY WARNING =================
with tab_flood:
    st.subheader("When will the basin flood? (Dynamic EWS)")
    st.markdown("Spatial susceptibility multiplied by the Himalayan-pooled temporal trigger model.")

    main1, main2 = st.columns([1, 1.2])
    with main1:
        if temporal_prob > THRESHOLD and burst:
            st.markdown(f'<div class="status-card status-alert"><h3>HIGH RISK ALERT</h3><p>Flood probability ({temporal_prob*100:.1f}%) exceeds threshold with an active rainfall burst.</p></div>', unsafe_allow_html=True)
            st.markdown("**Action:** Issue warnings; pre-position rescue in red zones.")
        elif temporal_prob > THRESHOLD:
            st.markdown(f'<div class="status-card status-watch"><h3>SATURATED BASIN WATCH</h3><p>Flood probability ({temporal_prob*100:.1f}%) is high due to antecedent saturation, but no burst detected.</p></div>', unsafe_allow_html=True)
            st.markdown("**Action:** Primed state. Monitor for forecast bursts.")
        elif temporal_prob > 0.05:
            st.markdown(f'<div class="status-card status-watch"><h3>ELEVATED WATCH</h3><p>Flood probability ({temporal_prob*100:.1f}%) is slightly elevated.</p></div>', unsafe_allow_html=True)
            st.markdown("**Action:** Routine monitoring.")
        else:
            st.markdown(f'<div class="status-card status-safe"><h3>NORMAL OPERATIONS</h3><p>Basin flood risk ({temporal_prob*100:.1f}%) is within safe limits.</p></div>', unsafe_allow_html=True)
            st.markdown("**Action:** Standard operations.")

        st.markdown("#### Key Indicators")
        m1, m2 = st.columns(2)
        m1.metric("Basin Flood Risk", f"{temporal_prob*100:.1f}%")
        m2.metric("Threshold Limit", f"{THRESHOLD*100:.0f}%")
        with st.expander("Methodology & Data Sources"):
            st.markdown("""
            **Spatial Layer:** Random Forest (AUC 0.984) — GFD inventory + HAND hydrology.
            **Temporal Layer:** Himalayan Pooled XGBoost (AUC 0.750) — four regions, 42-year climatologically normalized features.
            **Event Detection:** 85.7% recall on out-of-sample 2019–2023 events. **Lead Time:** 6 days.
            """)

    with main2:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
        im = draw_map(ax, flood_map, flood_bounds, raster_crs, flood_map * temporal_prob, blues, 0.02)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Dynamic Flood Risk (0.0–1.0)", rotation=270, labelpad=18)
        st.pyplot(fig)

# ================= TAB 2: LANDSLIDE HAZARD =================
with tab_ls:
    st.subheader("Landslide susceptibility + trigger advisory")
    st.markdown("Static susceptibility (Where) with a rule-based meteorological trigger advisory (When).")

    if ls_map is None:
        st.error("Landslide susceptibility raster not found.")
    else:
        main1, main2 = st.columns([1, 1.2])
        with main1:
            trigger_score = 0.0
            if r7 >= 40: trigger_score += 0.5
            if max3 >= 30: trigger_score += 0.3
            if (melt >= 10) and (snow >= 20): trigger_score += 0.2

            if trigger_score >= 0.5:
                st.markdown('<div class="status-card status-alert"><h3>TRIGGER CONDITIONS PRESENT</h3><p>Intense rainfall and/or active snowmelt over high-susceptibility slopes.</p></div>', unsafe_allow_html=True)
                st.markdown("**Action:** Restrict movement on steep slopes and known corridors.")
            elif trigger_score >= 0.3:
                st.markdown('<div class="status-card status-watch"><h3>ELEVATED WATCH</h3><p>Partial trigger conditions present over susceptible terrain.</p></div>', unsafe_allow_html=True)
                st.markdown("**Action:** Increase patrol of vulnerable road sections.")
            else:
                st.markdown('<div class="status-card status-safe"><h3>NORMAL OPERATIONS</h3><p>No significant rainfall or snowmelt trigger.</p></div>', unsafe_allow_html=True)
                st.markdown("**Action:** Standard monitoring.")

            st.markdown("#### Key Indicators")
            m3, m4 = st.columns(2)
            m3.metric("Static Model AUC", "0.938")
            m4.metric("Trigger Type", "Rule-Based")

            st.markdown('<div class="info-box"><b>Data Limitation:</b> A temporal ML landslide model could not be validated because the hydro-meteorological catalog contains zero events after 2018 (documentation gap, not hazard absence). This layer uses empirical rainfall/snow thresholds as the operational fallback.</div>', unsafe_allow_html=True)

        with main2:
            dynamic_ls = np.clip(ls_map * (1.0 + trigger_score * 2.0), 0, 1)
            fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
            im = draw_map(ax, ls_map, ls_bounds, raster_crs, dynamic_ls, reds, 0.1)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label("Active Landslide Risk (0.0–1.0)", rotation=270, labelpad=18)
            st.pyplot(fig)

# --- FOOTER ---
st.markdown("---")
f1, f2 = st.columns([2, 1])
with f1:
    st.caption("Operational Multi-Hazard EWS Prototype | Powered by CHIRPS, MODIS, and Machine Learning")
with f2:
    st.caption("**Developed by:** Tazmeen Zargar © | 2026")