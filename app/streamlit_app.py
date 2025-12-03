
import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

# with open("style.css") as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(
    page_title="Camera privacy risk heatmap",
    layout="wide"
)

st.title("Camera privacy risk heatmap")
st.write(
    "This app visualizes **privacy risk** for several popular smart-camera devices "
    "based on the approximate **face pixel height** they can capture at different distances. "
    "Higher face pixel heights mean it is more likely the camera can generate a usable facial "
    "embedding for recognition systems (e.g., ArcFace-style models)."
)

DATA_PATH = Path(__file__).parent / "data" / "devices.json"
with open(DATA_PATH, "r") as f:
    devices = json.load(f)

df_devices = pd.DataFrame(devices)



# Sidebar controls
st.sidebar.header("Controls")

face_height_m = st.sidebar.slider(
    "Assumed Face Height (meters)",
    min_value=0.16,
    max_value=0.24,
    value=0.20,
    step=0.01,
    help="Approximate vertical size of a human face."
)

# min_distance = st.sidebar.slider(
#     "Minimum Distance to Plot (m)",
#     min_value=0.5,
#     max_value=5.0,
#     value=0.5,
#     step=0.5,
# )

min_distance = 0.5   # MINIMUM DISTANCE 0.5 TO AVOID ZERO DIVISION

max_distance = st.sidebar.slider(
    "Maximum Distance to Plot (m)",
    min_value=5.0,
    max_value=15.0,
    value=10.0,
    step=0.5,
    help="Maximum limit of the heatmap x-axis."
)

# distance_step = st.sidebar.select_slider(
#     "Distance Step (m)",
#     options=[0.25, 0.5, 1.0],
#     value=0.5,
# )

distance_step = 0.25

# --- DEVICE SELECTION MODE ---
st.sidebar.header("Device Options")

add_custom_device = st.sidebar.checkbox(
    "Add a custom device",
    value=False,
    help="Enable this if you want to add a user-defined camera to the analysis."
)

custom_device_df = pd.DataFrame()

if add_custom_device:
    st.sidebar.markdown("### Custom Device Parameters")

    custom_name = st.sidebar.text_input(
        "Device name",
        value="User Custom Device"
    )

    custom_resolution_v = st.sidebar.number_input(
        "Vertical Resolution (pixels)",
        min_value=100,
        max_value=8000,
        value=1080,
        step=10
    )

    custom_fov_v = st.sidebar.number_input(
        "Vertical FOV (degrees)",
        min_value=5.0,
        max_value=170.0,
        value=90.0,
        step=1.0
    )

    custom_min_dist = st.sidebar.number_input(
        "Suggested Min Distance (m)",
        min_value=0.0,
        max_value=20.0,
        value=0.5,
        step=0.1
    )

    custom_max_dist = st.sidebar.number_input(
        "Suggested Max Distance (m)",
        min_value=0.5,
        max_value=50.0,
        value=10.0,
        step=0.5
    )

    custom_device_df = pd.DataFrame([{
        "device": custom_name,
        "brand": "Custom",
        "form_factor": "user_defined",
        "resolution_v": custom_resolution_v,
        "fov_v_deg": custom_fov_v,
        "min_dist_m": custom_min_dist,
        "max_dist_m": custom_max_dist
    }])

# Append custom device (if exists) to the main device list
df_devices_all = pd.concat([df_devices, custom_device_df], ignore_index=True)

device_filter = st.sidebar.multiselect(
    "Filter by Brand",
    options=sorted(df_devices["brand"].unique()),
    default=list(sorted(df_devices["brand"].unique()))
)

form_factor_filter = st.sidebar.multiselect(
    "Filter by Form Factor",
    options=sorted(df_devices["form_factor"].unique()),
    default=list(sorted(df_devices["form_factor"].unique()))
)

# Risk thresholds (in face pixels)
st.sidebar.markdown("### Risk Thresholds (Face Height in Pixels)")
# --- Threshold Inputs with Enforced Valid Ranges ---

# HIGH risk threshold (top of hierarchy)
high_thresh = st.sidebar.number_input(
    "High Risk ≥ (px)",
    min_value=112,
    max_value=1000,
    value=200,
    step=5,
    help="Faces with pixel height ≥ this value are considered high privacy risk."
)

# MEDIUM risk threshold (must be strictly below high_thresh)
usable_thresh = st.sidebar.number_input(
    "Medium Risk ≥ (px)",
    min_value=80,
    max_value=high_thresh - 1,
    value=112,
    step=1,
    help=f"Must be strictly less than the High Risk threshold ({high_thresh})."
)

# LOW risk threshold (must be strictly below usable_thresh)
weak_thresh = st.sidebar.number_input(
    "Low Risk ≥ (px)",
    min_value=40,
    max_value=usable_thresh - 1,
    value=80,
    step=1,
    help=f"Must be strictly less than the Medium Risk threshold ({usable_thresh})."
)

# MINIMAL risk threshold (must be strictly below weak_thresh)
minimal_thresh = st.sidebar.number_input(
    "Minimal Risk ≥ (px)",
    min_value=1,
    max_value=weak_thresh - 1,
    value=40,
    step=1,
    help=f"Must be strictly less than the Weak Risk threshold ({weak_thresh})."
)

def compute_face_px(resolution_v, fov_v_deg, distance_m, face_height_m):
    """Compute face pixel height using simple geometric projection."""
    theta = np.deg2rad(fov_v_deg)
    scene_height_m = 2 * distance_m * np.tan(theta / 2)
    if scene_height_m <= 0:
        return np.nan
    m_per_px = scene_height_m / resolution_v
    return face_height_m / m_per_px

# def classify_risk(face_px):
#     if np.isnan(face_px):
#         return "unknown", 0
#     if face_px >= high_thresh:
#         return "high", 3
#     elif face_px >= usable_thresh:
#         return "medium", 2
#     elif face_px >= weak_thresh:
#         return "low", 1
#     else:
#         return "minimal", 0

def classify_risk(face_px):
    """
    Returns (risk_label, risk_score)
    risk_score drives the color scale.
    """
    if np.isnan(face_px):
        return "unknown", 0
    # 5-level classification
    if face_px >= high_thresh:
        return "ultrahigh", 4
    if usable_thresh <= face_px < high_thresh:
        return "high", 3
    if weak_thresh <= face_px < usable_thresh:
        return "medium", 2
    if minimal_thresh <= face_px < weak_thresh:
        return "low", 1

    # New category: extremely small face region (<40 px)
    return "minimal", 0


df_devices_f = df_devices_all[
    df_devices_all["brand"].isin(device_filter + (["Custom"] if add_custom_device else []))
    & df_devices_all["form_factor"].isin(form_factor_filter + (["user_defined"] if add_custom_device else []))
]


distances = np.arange(min_distance, max_distance + 1e-6, distance_step)

records = []
for _, row in df_devices_f.iterrows():
    for d in distances:
        face_px = compute_face_px(
            resolution_v=row["resolution_v"],
            fov_v_deg=row["fov_v_deg"],
            distance_m=d,
            face_height_m=face_height_m,
        )
        risk_label, risk_score = classify_risk(face_px)
        records.append({
            "device": row["device"],
            "brand": row["brand"],
            "form_factor": row["form_factor"],
            "distance_m": d,
            "face_px": face_px,
            "risk_label": risk_label,
            "risk_score": risk_score,
        })

df_heatmap = pd.DataFrame(records)

st.markdown("### Privacy risk heatmap")

heatmap = (
    alt.Chart(df_heatmap)
    .mark_rect()
    .encode(
        # Treat distance as discrete / ordinal so each distinct value gets its own column
        x=alt.X("distance_m:O",
                title="Distance from Camera (m)",
                axis=alt.Axis(
                    labelExpr="datum.value % 0.5 === 0 ? datum.label : ''"
                )
            ),
        y=alt.Y("device:N", title="Device"),
        color=alt.Color(
            "risk_score:Q",
            scale=alt.Scale(
                domain=[0, 1, 2, 3, 4],
                range=[
                    "#f0f0f0",   # 0 minimal
                    "#d9d9d9",   # 1 low
                    "#a6cee3",   # 2 medium
                    "#fdbf6f",   # 3 high
                    "#e31a1c"    # 4 ultrahigh
                    ]
            ),
            legend=alt.Legend(
                title="Privacy Risk Level",
                labelExpr="{0:'minimal',1:'low',2:'medium',3:'high',4:'ultrahigh'}[datum.value]"
            ),
        ),
        tooltip=[
            alt.Tooltip("device:N"),
            alt.Tooltip("brand:N"),
            alt.Tooltip("distance_m:Q", format=".2f"),
            alt.Tooltip("face_px:Q", format=".1f", title="Face Height (px)"),
            alt.Tooltip("risk_label:N", title="Risk Category"),
        ],
    )
    .properties(
        height=300,
        width=700,
    )
)

st.altair_chart(heatmap, use_container_width=True)
st.caption(
    "Note: Distance starts at 0.5 m for visualization to avoid divide-by-zero issues in the geometric projection formula."
)
st.markdown(
    """
    **Interpretation**

    - **Ultra-high risk**: face pixel height ≥ high threshold (default 200 px). Conditions are very likely to
      support high-quality facial embeddings for modern recognition models.
    - **High risk**: between usable and high thresholds (default 112–199 px). Embeddings are likely usable but
      more sensitive to pose, occlusion, and lighting.
    - **Medium risk**: between weak and usable thresholds (default 80–111 px). Recognition is unstable; embeddings
      may fail frequently in real-world conditions.
    - **Low risk**: below weak threshold (default 40-79 px). Faces may be detectable but typically insufficient
      for robust recognition; below ~40 px, even detection often fails.
    - **Minimal/ultra-low risk**: below ~40 px, even detection can fails (although TinyFace uses 20 by 16 px)

    These bands loosely reflect findings from NIST FRVT, ArcFace training resolutions, and small-face detection
    benchmarks (TinyFace, etc).
    """
)

st.markdown("### Device Overview")
st.dataframe(
    df_devices_f[[
        "device", "brand", "form_factor",
        "resolution_v", "fov_v_deg", "min_dist_m", "max_dist_m"
    ]].rename(columns={
        "resolution_v": "vertical_resolution_px",
        "fov_v_deg": "vertical_fov_deg"
    }),
    use_container_width=True
)

st.markdown("### Face-Resolution Formula")
st.latex(
    r"\text{face}_{px} = \frac{R_v \cdot H_f}{2 D \tan\left(\frac{\theta}{2}\right)}"
)
st.markdown(
    """
    where:

    - $R_v$ = vertical resolution of the camera (pixels)  
    - $H_f$ = real-world face height (meters)  
    - $D$ = distance from camera to subject (meters)  
    - $theta$ = vertical field of view (radians)
    """
)


st.markdown(

    """
    ### Device Variable Definitions

    Each camera device in the analysis is described using the following parameters:

    - **`device`** — *Device name*  
    The commercial product identifier (e.g., “Ring Video Doorbell”, “Eufy 2K Doorbell”).

    - **`brand`** — *Manufacturer name*  
    Useful for filtering the heatmap (Ring, Nest, Wyze, Arlo, Eufy).

    - **`form_factor`** — *Device category*  
    Indicates typical usage and mounting distance:  
    - `doorbell` (0.5–2 m)  
    - `floodlight` (3–8 m)  
    - `outdoor_cam` (3–12 m)  
    - `indoor_outdoor_cam` (2–8 m)

    - **`resolution_v`** — *Vertical resolution (pixels)*  
    The height of the video frame in pixel units. Examples:  
    - 1080p → `resolution_v = 1080`  
    - 4K → `resolution_v = 2160`  
    - 2K portrait doorbells → `resolution_v = 1536`  
    Higher values increase facial detail and recognition risk.

    - **`fov_v_deg`** — *Vertical field of view (degrees)*  
    The vertical angular span of the camera’s imaging area.  
    Narrower FOV concentrates pixels → higher pixel density on faces.  
    Wider FOV spreads pixels → lower face detail.

    - **`min_dist_m`** & **`max_dist_m`** — *Typical operating distance range (meters)*  
    These describe the realistic physical distances at which people appear in front of the camera.  
    They are **not constraints in the formula** but provide useful context and defaults for plotting.

    """
)
