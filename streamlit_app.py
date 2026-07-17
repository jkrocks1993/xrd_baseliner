#!/usr/bin/env python3
"""
XRD Data Analyzer
=================
A Streamlit web app for baseline correction and 2θ peak detection/measurement 
on powder X-ray diffraction (XRD) data.

Features:
- Upload common XRD export files (CSV, TXT, DAT, XY, XLSX)
- Robust parsing with options for headers, delimiters, column selection
- Multiple baseline correction methods (via pybaselines)
- Smoothing (Savitzky-Golay)
- Interactive peak detection with adjustable parameters
- Automatic d-spacing calculation (user-selectable wavelength)
- Interactive Plotly plots (raw, baseline, corrected, peaks)
- Export processed data and peak list as CSV
- Demo data generator for testing

Installation (run once):
    pip install -r requirements.txt

Run:
    streamlit run xrd_analyzer.py

Author: Grok-assisted development for scientific use
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, savgol_filter, detrend
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import io
import warnings
warnings.filterwarnings("ignore")

# Try to import pybaselines (recommended for high-quality baseline methods)
try:
    from pybaselines import Baseline
    HAS_PYBASELINES = True
except ImportError:
    HAS_PYBASELINES = False


# ====================== 3D CRYSTAL VISUALIZATION (Plotly) ======================
ELEMENT_COLORS = {
    'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00',
    'B': '#FFB5B5', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
    'F': '#90E050', 'Ne': '#B3E3F5', 'Na': '#AB5CF2', 'Mg': '#8AFF00',
    'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000', 'S': '#FFFF30',
    'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
    'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB', 'Cr': '#8A99C7',
    'Mn': '#9C7AC7', 'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050',
    'Cu': '#C88033', 'Zn': '#7D80B0', 'Ga': '#C28F8F', 'Ge': '#668F8F',
    'As': '#BD80E3', 'Se': '#FFA100', 'Br': '#A62929', 'Kr': '#5CB8D1',
    'Rb': '#702EB0', 'Sr': '#00FF00', 'Y': '#94FFFF', 'Zr': '#94E0E0',
    'Nb': '#73C2C9', 'Mo': '#54B5B5', 'Tc': '#3B9E9E', 'Ru': '#248F8F',
    'Rh': '#0A7D8C', 'Pd': '#006985', 'Ag': '#C0C0C0', 'Cd': '#FFD98F',
    'In': '#A67573', 'Sn': '#668080', 'Sb': '#9E63B5', 'Te': '#D47A00',
    'I': '#940094', 'Xe': '#429EB2', 'Cs': '#57178F', 'Ba': '#00C900',
    'La': '#70D4FF', 'Ce': '#FFFFC7', 'Pr': '#D9FFC7', 'Nd': '#C7FFC7',
    'Pm': '#A3FFC7', 'Sm': '#8FFFC7', 'Eu': '#61FFC7', 'Gd': '#45FFC7',
    'Tb': '#30FFC7', 'Dy': '#1FFFC7', 'Ho': '#00FF9C', 'Er': '#00E675',
    'Tm': '#00D452', 'Yb': '#00BF38', 'Lu': '#00AB24', 'Hf': '#4DC2FF',
    'Ta': '#4DA6FF', 'W': '#2194D6', 'Re': '#267DAB', 'Os': '#266696',
    'Ir': '#175487', 'Pt': '#D0D0E0', 'Au': '#FFD123', 'Hg': '#B8B8D0',
    'Tl': '#A6544D', 'Pb': '#575961', 'Bi': '#9E4FB5', 'Po': '#AB5C00',
    'At': '#754F45', 'Rn': '#428296', 'Fr': '#420066', 'Ra': '#007D00',
    'Ac': '#70ABFA', 'Th': '#00BAFF', 'Pa': '#00A1FF', 'U': '#008FFF',
    'Np': '#0080FF', 'Pu': '#006BFF', 'Am': '#545CF2', 'Cm': '#785CE3',
    'Bk': '#8A4FE3', 'Cf': '#A136D4', 'Es': '#B31FD4', 'Fm': '#B31FBA',
}

def get_element_color(element):
    return ELEMENT_COLORS.get(element, '#CCCCCC')  # Default gray


def create_crystal_3d_plot(structure, title="Crystal Structure"):
    """Create an interactive 3D Plotly visualization of a pymatgen Structure."""
    if structure is None:
        return None

    # Get fractional coordinates and convert to cartesian
    coords = structure.cart_coords
    elements = [site.specie.symbol for site in structure]

    # Group by element for legend
    from collections import defaultdict
    element_groups = defaultdict(list)

    for i, (coord, elem) in enumerate(zip(coords, elements)):
        element_groups[elem].append(coord)

    fig = go.Figure()

    for elem, positions in element_groups.items():
        x, y, z = zip(*positions)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=8,
                color=get_element_color(elem),
                line=dict(width=0.5, color='black')
            ),
            name=elem,
            legendgroup=elem,
            showlegend=True,
            hovertemplate=f"<b>{elem}</b><br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>"
        ))

    # Add unit cell edges (simple box)
    cell = structure.lattice.matrix
    origin = np.array([0, 0, 0])
    vertices = [
        origin,
        cell[0],
        cell[1],
        cell[2],
        cell[0] + cell[1],
        cell[0] + cell[2],
        cell[1] + cell[2],
        cell[0] + cell[1] + cell[2]
    ]
    edges = [
        (0,1), (0,2), (0,3),
        (1,4), (1,5), (2,4), (2,6), (3,5), (3,6),
        (4,7), (5,7), (6,7)
    ]

    for start, end in edges:
        fig.add_trace(go.Scatter3d(
            x=[vertices[start][0], vertices[end][0]],
            y=[vertices[start][1], vertices[end][1]],
            z=[vertices[start][2], vertices[end][2]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='data'
        ),
        legend_title_text="Elements",
        height=550,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


# ====================== STREAMLIT APP ======================

st.set_page_config(
    page_title="XRD Peak Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for nicer look
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }
    .stMetric { background-color: #f0f2f6; border-radius: 8px; padding: 8px; }
    .peak-table { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("🧪 XRD Data Analyzer")
st.caption("**Measure 2θ values • Baseline correction • Peak detection** for powder XRD patterns")

# ====================== SIDEBAR CONTROLS ======================
with st.sidebar:
    st.header("⚙️ Controls")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload XRD data file",
        type=["csv", "txt", "dat", "xy", "xlsx", "xls"],
        help="Common formats from Bruker, Rigaku, PANalytical, etc. Export as ASCII/CSV if possible."
    )
    
    st.divider()
    
    # Demo data
    if st.button("📊 Load Demo XRD Data", use_container_width=True):
        st.session_state["use_demo"] = True
        st.rerun()
    
    if st.button("🔄 Reset App", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.divider()
    
    # Advanced file parsing
    with st.expander("📁 File Parsing Options", expanded=False):
        skip_rows = st.number_input("Skip first N rows (header/metadata)", min_value=0, max_value=50, value=0, step=1)
        delimiter = st.selectbox(
            "Delimiter",
            options=["Auto (recommended)", "Comma (,)", "Tab (\\t)", "Semicolon (;)", "Space", "Pipe (|)"],
            index=0
        )
        header_option = st.selectbox(
            "Header row",
            options=["Infer from file", "No header (use column names below)", "First row is header"],
            index=0
        )
    
    st.divider()
    
    # Preprocessing
    st.subheader("Preprocessing")
    
    do_smoothing = st.checkbox("Apply Savitzky-Golay smoothing", value=True)
    if do_smoothing:
        sg_window = st.slider("Smoothing window length (odd)", min_value=5, max_value=101, value=21, step=2)
        sg_poly = st.slider("Polynomial order", min_value=1, max_value=5, value=3)
    
    st.divider()
    
    # Baseline correction
    st.subheader("Baseline Correction")
    
    if HAS_PYBASELINES:
        baseline_method = st.selectbox(
            "Method (pybaselines)",
            options=[
                "als (Asymmetric Least Squares)",
                "airpls (Adaptive Iteratively Reweighted PLS)",
                "arpls (Asymmetrically Reweighted PLS)",
                "aspls (Adaptive Smoothness PLS)",
                "drpls (Doubly Reweighted PLS)",
                "iasls (Improved AsLS)",
                "snip (Statistics-sensitive Non-linear Iterative Peak-clipping)",
                "rolling_ball",
                "polynomial",
                "modpoly (Modified Polynomial)",
                "imodpoly (Improved Modified Poly)",
                "rubberband",
                "None (no correction)"
            ],
            index=0,
            help="ALS / airPLS / ARPLS / asPLS work well for curved XRD backgrounds. SNIP is a classic algorithm used in many XRD/XRF packages. Rolling ball is simple and robust. Polynomial methods suit gentle curves."
        )
    else:
        st.warning("⚠️ pybaselines not installed. Using built-in ALS only.\n\nInstall with: `pip install pybaselines` for more methods.")
        baseline_method = st.selectbox(
            "Method",
            options=["als (built-in)", "linear detrend", "None"],
            index=0
        )
    
    # Method-specific parameters
    method_lower = baseline_method.lower()
    
    # Whittaker-style methods (need λ, sometimes p and iterations)
    if any(k in method_lower for k in ["als", "airpls", "arpls", "aspls", "drpls", "iasls"]):
        default_lam = 1e6 if "airpls" in method_lower else 1e5
        lam = st.number_input(
            "λ (smoothness, higher = smoother baseline)",
            value=default_lam, min_value=1e2, max_value=1e10, step=1e4, format="%.0e",
            help="Typical XRD range: 1e4 – 1e7. Increase if baseline is too wiggly."
        )
        if "als" in method_lower or "iasls" in method_lower:
            p = st.slider(
                "p (asymmetry, lower = more baseline below data)",
                min_value=0.001, max_value=0.5, value=0.01, step=0.001, format="%.3f"
            )
        if any(k in method_lower for k in ["als", "airpls", "iasls"]):
            niter = st.slider(
                "Max iterations",
                min_value=5, max_value=100,
                value=50 if "airpls" in method_lower else 10
            )
    
    # SNIP parameters
    elif "snip" in method_lower:
        snip_half_window = st.slider(
            "Max half-window (iterations)",
            min_value=5, max_value=200, value=40, step=1,
            help="Controls how broad features can be. Start ~20–60 for typical powder XRD."
        )
        snip_decreasing = st.checkbox("Decreasing window order (usually smoother)", value=True)
        snip_smooth = st.slider("Smoothing half-window (0 = off)", min_value=0, max_value=20, value=3)
    
    # Rolling ball
    elif "rolling_ball" in method_lower:
        ball_hw = st.slider(
            "Half-window size",
            min_value=3, max_value=150, value=30, step=1,
            help="Related to the width of features you want to keep as peaks (not baseline)."
        )
    
    # Polynomial family
    elif any(k in method_lower for k in ["poly", "rubberband"]):
        poly_degree = st.slider("Polynomial degree", min_value=1, max_value=12, value=4)
    
    st.divider()
    
    # Peak detection
    st.subheader("Peak Detection")
    
    height_mode = st.radio("Height threshold type", ["Absolute", "% of max intensity"], horizontal=True, index=1)
    
    if height_mode == "% of max intensity":
        height_pct = st.slider("Min peak height (% of max)", min_value=0.5, max_value=50.0, value=5.0, step=0.5)
    else:
        height_abs = st.number_input("Min peak height (absolute)", value=100.0, min_value=0.0)
    
    prominence = st.slider("Prominence (peak distinctness)", min_value=0.1, max_value=100.0, value=10.0, step=0.5,
                           help="Higher = only sharp, prominent peaks. Lower = more peaks including shoulders.")
    min_distance = st.slider("Min distance between peaks (degrees 2θ)", min_value=0.05, max_value=5.0, value=0.3, step=0.05)
    min_width = st.slider("Min peak width (degrees 2θ)", min_value=0.01, max_value=2.0, value=0.1, step=0.01)
    
    st.divider()
    
    # Wavelength for d-spacing
    st.subheader("d-spacing Calculation")
    wavelength_option = st.selectbox(
        "X-ray source / Wavelength (Å)",
        options=["Cu Kα (1.5406)", "Cu Kα1 (1.54056)", "Co Kα (1.7890)", "Mo Kα (0.7107)", "Cr Kα (2.2897)", "Custom"],
        index=0
    )
    if wavelength_option == "Custom":
        wavelength = st.number_input("Wavelength (Å)", value=1.5406, min_value=0.1, max_value=10.0, step=0.0001, format="%.4f")
    else:
        wavelength = float(wavelength_option.split("(")[1].split(")")[0])
    
    st.caption(f"Using λ = {wavelength:.4f} Å → d = λ / (2 sin θ)")

# ====================== MAIN AREA ======================

def parse_xrd_file(file, skip_rows, delimiter, header_option):
    """Robust parser for various XRD export formats."""
    try:
        sep_map = {
            "Auto (recommended)": None,
            "Comma (,)": ",",
            "Tab (\\t)": "\t",
            "Semicolon (;)": ";",
            "Space": r"\s+",
            "Pipe (|)": "|"
        }
        sep = sep_map.get(delimiter, None)
        
        if header_option == "No header (use column names below)":
            header = None
        elif header_option == "First row is header":
            header = 0
        else:
            header = "infer"
        
        if file.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file, header=header, skiprows=skip_rows)
        else:
            df = pd.read_csv(
                file,
                sep=sep,
                header=header,
                skiprows=skip_rows,
                engine="python",
                on_bad_lines="skip"
            )
        
        # Clean column names
        df.columns = [str(c).strip().replace("#", "").replace("(", "").replace(")", "") for c in df.columns]
        
        # If only 2 columns and no good names, rename
        if len(df.columns) == 2 and (df.columns[0].lower() in ["0", "unnamed: 0", ""] or pd.api.types.is_numeric_dtype(df.iloc[0, 0])):
            df.columns = ["2theta", "Intensity"]
        
        return df, None
    except Exception as e:
        return None, str(e)

def generate_demo_data(n_points=2500):
    """Generate realistic synthetic XRD pattern for testing."""
    np.random.seed(42)
    x = np.linspace(5, 80, n_points)
    
    # Realistic curved background (slowly varying)
    baseline = 120 + 2.5 * x - 0.015 * x**2 + 0.00008 * x**3
    
    # Typical peaks for e.g. a mixed phase sample (positions in 2θ)
    peak_params = [
        (11.2, 850, 0.25),
        (18.7, 620, 0.22),
        (23.4, 1200, 0.18),
        (29.1, 480, 0.30),
        (33.8, 950, 0.20),
        (38.5, 310, 0.35),
        (42.2, 780, 0.17),
        (48.9, 550, 0.28),
        (55.3, 420, 0.24),
        (61.7, 680, 0.19),
        (67.4, 290, 0.32),
        (73.1, 380, 0.21),
    ]
    
    y = baseline.copy()
    for pos, height, fwhm in peak_params:
        # Lorentzian profile (common for XRD)
        gamma = fwhm / 2
        y += height * (gamma**2 / ((x - pos)**2 + gamma**2))
    
    # Add realistic noise
    noise_level = np.sqrt(y) * 0.8 + 8
    y += np.random.normal(0, noise_level)
    
    # Make sure positive
    y = np.maximum(y, 0)
    
    df = pd.DataFrame({"2theta": x, "Intensity": y})
    true_peaks = [p[0] for p in peak_params]
    return df, true_peaks

# Handle demo data
if "use_demo" in st.session_state and st.session_state["use_demo"]:
    df_raw, true_peaks_demo = generate_demo_data()
    st.session_state["df_raw"] = df_raw
    st.session_state["use_demo"] = False
    st.info("✅ Demo data loaded! It contains 12 synthetic peaks on a curved background. Try different baseline methods.")

# Process uploaded or demo data
df_raw = st.session_state.get("df_raw", None)

if uploaded_file is not None:
    df_raw, error = parse_xrd_file(uploaded_file, skip_rows, delimiter, header_option)
    if error:
        st.error(f"Failed to parse file: {error}")
        st.stop()
    st.session_state["df_raw"] = df_raw
    st.success(f"✅ File loaded: **{uploaded_file.name}** ({len(df_raw)} rows)")

if df_raw is None:
    st.info("👆 Upload an XRD data file above, or click **Load Demo XRD Data** to try the app.")
    st.stop()

# Column selection
st.subheader("Select Data Columns")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    x_col = st.selectbox(
        "2θ column (angle in degrees)",
        options=df_raw.columns.tolist(),
        index=0 if "2theta" in str(df_raw.columns[0]).lower() or "theta" in str(df_raw.columns[0]).lower() else 0
    )
with col2:
    y_col = st.selectbox(
        "Intensity column",
        options=df_raw.columns.tolist(),
        index=1 if len(df_raw.columns) > 1 else 0
    )
with col3:
    st.metric("Data points", f"{len(df_raw):,}")

# Extract and clean data
x = pd.to_numeric(df_raw[x_col], errors="coerce").to_numpy()
y = pd.to_numeric(df_raw[y_col], errors="coerce").to_numpy()

# Remove NaNs and sort
mask = np.isfinite(x) & np.isfinite(y)
x, y = x[mask], y[mask]
if len(x) == 0:
    st.error("No valid numeric data after cleaning. Check your column selection.")
    st.stop()

sort_idx = np.argsort(x)
x, y = x[sort_idx], y[sort_idx]

# Show raw data preview
with st.expander("🔍 Raw Data Preview (first 10 rows)", expanded=False):
    preview_df = pd.DataFrame({x_col: x[:10], y_col: y[:10]})
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

# ====================== PREPROCESSING ======================

y_proc = y.copy()

# 1. Smoothing
if do_smoothing:
    try:
        y_proc = savgol_filter(y_proc, window_length=sg_window, polyorder=sg_poly, mode="interp")
    except Exception as e:
        st.warning(f"Smoothing failed: {e}. Using raw data.")

# 2. Baseline correction
baseline = np.zeros_like(y_proc)

if "None" in baseline_method:
    baseline = np.zeros_like(y_proc)
elif HAS_PYBASELINES:
    baseline_obj = Baseline(x_data=x)
    method_key = baseline_method.split()[0].lower()
    
    try:
        if method_key == "als":
            baseline, _ = baseline_obj.asls(y_proc, lam=lam, p=p, max_iter=niter)
        elif method_key == "airpls":
            baseline, _ = baseline_obj.airpls(y_proc, lam=lam, max_iter=niter)
        elif method_key == "arpls":
            baseline, _ = baseline_obj.arpls(y_proc, lam=lam)
        elif method_key == "aspls":
            baseline, _ = baseline_obj.aspls(y_proc, lam=lam)
        elif method_key == "drpls":
            baseline, _ = baseline_obj.drpls(y_proc, lam=lam)
        elif method_key == "iasls":
            baseline, _ = baseline_obj.iasls(y_proc, lam=lam, p=p, max_iter=niter)
        elif method_key == "snip":
            baseline, _ = baseline_obj.snip(
                y_proc,
                max_half_window=snip_half_window,
                decreasing=snip_decreasing,
                smooth_half_window=snip_smooth if snip_smooth > 0 else None
            )
        elif method_key == "rolling_ball":
            baseline, _ = baseline_obj.rolling_ball(y_proc, half_window=ball_hw)
        elif method_key == "polynomial":
            baseline, _ = baseline_obj.polynomial(y_proc, poly_order=poly_degree)
        elif method_key == "modpoly":
            baseline, _ = baseline_obj.modpoly(y_proc, poly_order=poly_degree)
        elif method_key == "imodpoly":
            baseline, _ = baseline_obj.imodpoly(y_proc, poly_order=poly_degree)
        elif method_key == "rubberband":
            baseline, _ = baseline_obj.rubberband(y_proc, poly_order=poly_degree)
        else:
            baseline = np.zeros_like(y_proc)
    except Exception as e:
        st.error(f"Baseline correction failed with {baseline_method}: {e}")
        baseline = np.zeros_like(y_proc)
else:
    # Fallback built-in ALS
    if "als" in baseline_method.lower():
        try:
            L = len(y_proc)
            D = diags([1, -2, 1], [0, -1, 1], shape=(L-2, L))
            w = np.ones(L)
            for _ in range(niter):
                W = diags(w, 0, shape=(L, L))
                Z = W + lam * D.dot(D.transpose())
                z = spsolve(Z, w * y_proc)
                w = p * (y_proc > z) + (1 - p) * (y_proc < z)
            baseline = z
        except Exception as e:
            st.error(f"Built-in ALS failed: {e}")
            baseline = np.zeros_like(y_proc)
    elif "detrend" in baseline_method.lower():
        baseline = y_proc - detrend(y_proc, type="linear")
    else:
        baseline = np.zeros_like(y_proc)

y_corrected = y_proc - baseline

# ====================== PEAK DETECTION ======================

# Determine height threshold
if height_mode == "% of max intensity":
    height_threshold = (height_pct / 100.0) * np.max(y_corrected)
else:
    height_threshold = height_abs

# Find peaks
try:
    peaks, properties = find_peaks(
        y_corrected,
        height=height_threshold,
        prominence=prominence,
        distance=min_distance / np.mean(np.diff(x)),  # convert degrees to index distance
        width=min_width / np.mean(np.diff(x))
    )
except Exception as e:
    st.error(f"Peak detection failed: {e}")
    peaks = np.array([])

# Calculate d-spacing
theta_rad = np.deg2rad(x[peaks] / 2.0)
d_spacing = wavelength / (2 * np.sin(theta_rad)) if len(peaks) > 0 else np.array([])

# Build peaks dataframe
if len(peaks) > 0:
    peak_data = {
        "2θ (°)": np.round(x[peaks], 4),
        "d-spacing (Å)": np.round(d_spacing, 4),
        "Intensity (corr.)": np.round(y_corrected[peaks], 1),
        "Intensity (raw)": np.round(y[peaks], 1),
        "Prominence": np.round(properties["prominences"], 2),
        "Width (°)": np.round(properties["widths"] * np.mean(np.diff(x)), 3),
        "Height": np.round(properties["peak_heights"], 1),
    }
    peaks_df = pd.DataFrame(peak_data)
    peaks_df = peaks_df.sort_values("2θ (°)").reset_index(drop=True)
else:
    peaks_df = pd.DataFrame(columns=["2θ (°)", "d-spacing (Å)", "Intensity (corr.)", "Intensity (raw)", "Prominence", "Width (°)", "Height"])

# ====================== PLOTTING ======================

st.subheader("📈 Interactive XRD Pattern")

# Create figure with subplots for clarity
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.65, 0.35],
    subplot_titles=("Full Pattern (Raw + Baseline + Corrected + Peaks)", "Baseline-subtracted (used for peak detection)")
)

# Trace 1: Raw data
fig.add_trace(
    go.Scatter(
        x=x, y=y,
        mode="lines",
        name="Raw data",
        line=dict(color="#1f77b4", width=1.5),
        hovertemplate="2θ: %{x:.3f}°<br>Intensity: %{y:.1f}<extra></extra>"
    ),
    row=1, col=1
)

# Trace 2: Baseline
if np.any(baseline != 0):
    fig.add_trace(
        go.Scatter(
            x=x, y=baseline,
            mode="lines",
            name="Baseline",
            line=dict(color="#d62728", width=2, dash="dash"),
            hovertemplate="2θ: %{x:.3f}°<br>Baseline: %{y:.1f}<extra></extra>"
        ),
        row=1, col=1
    )

# Trace 3: Corrected data
fig.add_trace(
    go.Scatter(
        x=x, y=y_corrected,
        mode="lines",
        name="Baseline-corrected",
        line=dict(color="#2ca02c", width=1.5),
        hovertemplate="2θ: %{x:.3f}°<br>Corr. Int.: %{y:.1f}<extra></extra>"
    ),
    row=1, col=1
)

# Peak markers on top plot
if len(peaks) > 0:
    fig.add_trace(
        go.Scatter(
            x=x[peaks],
            y=y_corrected[peaks],
            mode="markers",
            name=f"Detected peaks ({len(peaks)})",
            marker=dict(color="#ff7f0e", size=10, symbol="diamond", line=dict(color="white", width=1)),
            hovertemplate="Peak @ %{x:.4f}°<br>d = %{customdata[0]:.4f} Å<br>Int: %{y:.1f}<extra></extra>",
            customdata=np.stack([d_spacing], axis=-1) if len(d_spacing) > 0 else None
        ),
        row=1, col=1
    )

# Bottom plot: only corrected + peaks
fig.add_trace(
    go.Scatter(
        x=x, y=y_corrected,
        mode="lines",
        name="Corrected (detail)",
        line=dict(color="#2ca02c", width=1.5),
        showlegend=False,
        hovertemplate="2θ: %{x:.3f}°<br>Corr. Int.: %{y:.1f}<extra></extra>"
    ),
    row=2, col=1
)

if len(peaks) > 0:
    fig.add_trace(
        go.Scatter(
            x=x[peaks],
            y=y_corrected[peaks],
            mode="markers",
            name="Peaks",
            marker=dict(color="#ff7f0e", size=9, symbol="diamond"),
            showlegend=False,
            hovertemplate="Peak @ %{x:.4f}°<br>d = %{customdata[0]:.4f} Å<extra></extra>",
            customdata=np.stack([d_spacing], axis=-1) if len(d_spacing) > 0 else None
        ),
        row=2, col=1
    )

fig.update_xaxes(title_text="2θ (°)", row=2, col=1)
fig.update_yaxes(title_text="Intensity (a.u.)", row=1, col=1)
fig.update_yaxes(title_text="Corrected Intensity", row=2, col=1)

fig.update_layout(
    height=750,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=50, r=30, t=60, b=40)
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

# ====================== RESULTS ======================

col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.subheader(f"📋 Detected Peaks ({len(peaks_df)} found)")
    
    if len(peaks_df) > 0:
        st.dataframe(
            peaks_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "2θ (°)": st.column_config.NumberColumn(format="%.4f"),
                "d-spacing (Å)": st.column_config.NumberColumn(format="%.4f"),
            }
        )
        
        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Peaks detected", len(peaks_df))
        m2.metric("Strongest peak 2θ", f"{peaks_df['2θ (°)'].iloc[0]:.2f}°")
        m3.metric("d-spacing range", f"{peaks_df['d-spacing (Å)'].min():.2f} – {peaks_df['d-spacing (Å)'].max():.2f} Å")
    else:
        st.warning("No peaks detected with current settings. Try lowering prominence or height threshold.")

with col_right:
    st.subheader("💾 Export Results")
    
    # Processed data export
    processed_df = pd.DataFrame({
        "2theta_deg": x,
        "raw_intensity": y,
        "smoothed_intensity": y_proc if do_smoothing else y,
        "baseline": baseline,
        "baseline_corrected": y_corrected
    })
    
    csv_buffer = io.StringIO()
    processed_df.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="⬇️ Download Processed Data (CSV)",
        data=csv_buffer.getvalue(),
        file_name="xrd_processed_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    if len(peaks_df) > 0:
        peak_csv = io.StringIO()
        peaks_df.to_csv(peak_csv, index=False)
        
        st.download_button(
            label="⬇️ Download Peak List (CSV)",
            data=peak_csv.getvalue(),
            file_name="xrd_detected_peaks.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.caption("Processed file includes raw, smoothed, baseline, and corrected columns for further analysis in Origin, Excel, etc.")

# ====================== TIPS ======================
with st.expander("💡 Tips for Best Results & Common Issues", expanded=False):
    st.markdown("""
    **Baseline Correction:**
    - **ALS / airPLS / ARPLS / asPLS / drPLS**: Excellent for most powder XRD patterns with curved or sloping backgrounds. Increase λ for a smoother baseline; decrease *p* (when available) if the baseline is being pulled up into the peaks.
    - **SNIP**: Classic algorithm widely used in XRD/XRF software. Very effective for removing broad backgrounds while preserving peaks. Adjust the half-window size to match your peak widths.
    - **Rolling ball**: Simple morphological method. Robust and fast; good starting point when other methods overfit.
    - **Polynomial / modpoly / imodpoly**: Good when the background is gently curved. Higher degree = more flexible but can overfit peaks.
    - **Rubberband**: Connects local minima — useful for very spiky or noisy data.
    
    **Peak Detection:**
    - Start with **prominence ~ 5-15** and **height ~ 3-10%** of max.
    - If too many noise peaks: increase prominence and/or min distance.
    - If missing real peaks: lower prominence or height threshold.
    - The bottom subplot shows exactly what the algorithm "sees".
    
    **Data Quality:**
    - Smooth first if your pattern is noisy (window 11-31 typical for step size ~0.02°).
    - Always visually inspect the baseline fit before trusting peak positions.
    - For publication-quality positions, follow up with peak fitting (Gaussian/Lorentzian) in Origin or FullProf.
    
    **File Formats:**
    - Best: Export from instrument software as simple 2-column ASCII (2θ vs Intensity).
    - If your file has lots of metadata at the top, increase "Skip first N rows".
    
    **Limitations of this tool:**
    - Designed for **1D powder patterns**. For 2D detector images → use pyFAI first.
    - Peak positions are from `scipy.find_peaks`. For sub-pixel precision or full profile fitting, use lmfit or commercial software.
    """)

st.divider()

# ====================== PHASE IDENTIFICATION (Multi-Database) ======================
st.header("🔬 Phase Identification")

with st.expander("ℹ️ Note on JCPDS / ICDD vs Free Alternatives", expanded=False):
    st.markdown("""
    **Official JCPDS/ICDD Powder Diffraction File (PDF)** is a **commercial paid database** — the gold standard but requires a license.

    **Excellent free alternatives supported here**:
    - **Crystallography Open Database (COD)** — large open collection of *experimental* crystal structures (minerals, inorganics, organics). **No API key required.**
    - **AFLOW** — very large computational library + many ICSD-derived structures. **No API key required.**
    - **Materials Project** (high-quality computed + experimental structures; free API key at materialsproject.org)
    
    All three are combined with `pymatgen` for realistic XRD pattern simulation from actual crystal structures.
    You get crystal system, space group, lattice parameters, and a match score against your experimental peaks.
    """)

# Database selector
db_choice = st.radio(
    "Select database to search",
    options=[
        "Crystallography Open Database (COD) — free, no key",
        "AFLOW — free, no key (computational + ICSD)",
        "Materials Project (requires free API key)"
    ],
    index=0,
    horizontal=True,
    help="COD = experimental structures (great for minerals). AFLOW = large computational + experimental library. Materials Project = high-quality computed phases."
)

# Shared inputs
col1, col2 = st.columns([2, 1])
with col1:
    if "COD" in db_choice:
        search_query = st.text_input(
            "Chemical formula or elements",
            value="Fe2O3",
            help="Formula (Fe2O3, CaCO3) or elements. Commas or spaces both work: Fe, O  or  Fe O  or  Fe,O"
        )
    else:
        search_query = st.text_input(
            "Elements in your sample",
            value="Fe, O",
            help="Separate elements with commas or spaces. Examples: Fe, O   |   Fe O   |   Ca, C, O   |   Ti,O"
        )
with col2:
    max_cands = st.slider("Max candidates", min_value=5, max_value=30, value=12, step=1)

# ---------- Materials Project path ----------
mp_api_key = None
if "Materials Project" in db_choice:
    mp_api_key = st.text_input(
        "Materials Project API Key (get free at materialsproject.org)",
        type="password",
        placeholder="mp-XXXXXXXXXXXXXXXX",
        help="Your key is used only in the current browser session. It is never saved to disk or sent anywhere else."
    )

# Common imports needed for both
try:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.core import Structure, Composition
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    st.error("pymatgen is required. Install with: `pip install pymatgen`")

# ========== COD SEARCH ==========
if "COD" in db_choice and HAS_PYMATGEN:
    st.info("🔎 Searching the **Crystallography Open Database (COD)** — experimental structures, no API key needed.")

    if st.button("🔍 Search COD Database", type="primary", use_container_width=True):
        try:
            from pymatgen.ext.cod import COD
            import requests

            cod = COD(timeout=45)
            candidates = []

            query = search_query.strip()
            # Detect if it looks like a formula (contains digits) or element list
            is_formula_like = any(c.isdigit() for c in query) and "," not in query

            if is_formula_like:
                # Direct formula search (preferred for COD)
                try:
                    results = cod.get_structure_by_formula(query)
                    for item in results[:max_cands]:
                        struct = item["structure"]
                        candidates.append({
                            "id": f"cod-{item['cod_id']}",
                            "formula": struct.composition.reduced_formula,
                            "sg": item.get("sg", "N/A"),
                            "nsites": len(struct),
                            "structure": struct,
                            "source": "COD"
                        })
                except Exception as e:
                    st.warning(f"Formula search returned no/limited results ({e}). Trying element-based search...")

            if not candidates:
                # Element-based search via COD REST API (el1, el2, ...)
                elements = [e.strip() for e in query.replace(",", " ").split() if e.strip()]
                if not elements:
                    elements = [e.strip() for e in query.split(",") if e.strip()]

                params = {"format": "json"}
                for i, el in enumerate(elements[:8], start=1):
                    params[f"el{i}"] = el

                # Limit to reasonable number of distinct elements
                params["strictmax"] = len(elements) + 1  # allow a bit of flexibility

                resp = requests.get("https://www.crystallography.net/cod/result", params=params, timeout=45)
                resp.raise_for_status()
                entries = resp.json()

                # Take first N unique IDs and fetch structures
                seen = set()
                for entry in entries:
                    cod_id = int(entry.get("file", 0))
                    if cod_id in seen or cod_id == 0:
                        continue
                    seen.add(cod_id)
                    try:
                        struct = cod.get_structure_by_id(cod_id)
                        candidates.append({
                            "id": f"cod-{cod_id}",
                            "formula": struct.composition.reduced_formula,
                            "sg": entry.get("sg", "N/A"),
                            "nsites": len(struct),
                            "structure": struct,
                            "source": "COD"
                        })
                        if len(candidates) >= max_cands:
                            break
                    except Exception:
                        continue

            if candidates:
                st.session_state["phase_candidates"] = candidates
                st.session_state["phase_db"] = "COD"
                st.success(f"Found **{len(candidates)}** candidate structures from COD.")
            else:
                st.warning("No matching structures found in COD for this query. Try a simpler formula (e.g. Fe2O3) or fewer elements.")
        except Exception as e:
            st.error(f"COD search failed: {e}")
            st.info("Check your internet connection. COD is a public free service.")

# ========== AFLOW SEARCH ==========
elif "AFLOW" in db_choice and HAS_PYMATGEN:
    st.info("🔎 Searching **AFLOW** — large computational + ICSD-derived library (free, no API key).")

    # Optional dependency
    try:
        from aflow import search, K
        HAS_AFLOW = True
    except ImportError:
        HAS_AFLOW = False
        st.error("AFLOW Python client not installed. Run:\n`pip install aflow`")
        st.caption("The `aflow` package is a lightweight wrapper around the public AFLUX API.")

    if HAS_AFLOW:
        if st.button("🔍 Search AFLOW Database", type="primary", use_container_width=True):
            try:
                elements = [e.strip() for e in search_query.replace(",", " ").split() if e.strip()]
                if not elements:
                    st.warning("Please enter at least one element.")
                else:
                    # Build filter: species(Fe),species(O) or species(Fe,O) depending on version
                    # Most reliable: species(Fe),species(O),nspecies(2) for exact match, but we allow extra elements for flexibility
                    filter_parts = [f"species({el})" for el in elements]
                    # Prefer compounds that contain at least these elements
                    filter_str = ",".join(filter_parts)

                    # Use a reasonable batch and limit
                    results = search(batch_size=min(max_cands * 3, 50)).filter(filter_str)

                    candidates = []
                    count = 0
                    for entry in results:
                        if count >= max_cands:
                            break
                        try:
                            # Prefer getting a pymatgen Structure
                            structure = None

                            # Method 1: try ASE atoms → pymatgen (if ASE available)
                            try:
                                atoms = entry.atoms
                                if atoms is not None:
                                    structure = Structure.from_ase_atoms(atoms)
                            except Exception:
                                pass

                            # Method 2: fall back to downloading geometry / CONTCAR via aurl
                            if structure is None:
                                try:
                                    aurl = getattr(entry, "aurl", None) or entry.raw.get("aurl", "")
                                    if aurl:
                                        # AFLOW geometry endpoint returns POSCAR-like content
                                        import requests
                                        geom_url = aurl.rstrip("/") + "/?geometry"
                                        r = requests.get(geom_url, timeout=20)
                                        if r.status_code == 200 and r.text.strip():
                                            structure = Structure.from_str(r.text, fmt="poscar")
                                except Exception:
                                    pass

                            if structure is None:
                                continue

                            # Get some metadata
                            formula = getattr(entry, "compound", None) or structure.composition.reduced_formula
                            sg = getattr(entry, "spacegroup_relax", None) or "N/A"
                            try:
                                nsites = len(structure)
                            except Exception:
                                nsites = "?"

                            candidates.append({
                                "id": f"aflow-{getattr(entry, 'auid', count)}",
                                "formula": formula,
                                "sg": str(sg),
                                "nsites": nsites,
                                "structure": structure,
                                "source": "AFLOW",
                                "entry": entry  # keep reference if needed
                            })
                            count += 1
                        except Exception:
                            continue

                    if candidates:
                        st.session_state["phase_candidates"] = candidates
                        st.session_state["phase_db"] = "AFLOW"
                        st.success(f"Found **{len(candidates)}** candidate structures from AFLOW.")
                    else:
                        st.warning("No matching structures retrieved from AFLOW for this query. Try fewer elements or a common composition (e.g. Fe, O).")
            except Exception as e:
                st.error(f"AFLOW search failed: {e}")
                st.info("AFLOW servers can occasionally be slow or rate-limited. Try again in a moment.")

# ========== MATERIALS PROJECT SEARCH ==========
elif "Materials Project" in db_choice and HAS_PYMATGEN:
    if not mp_api_key:
        st.warning("Enter your free Materials Project API key above to search.")
    else:
        try:
            from mp_api.client import MPRester
            HAS_MP = True
        except ImportError:
            HAS_MP = False
            st.error("Missing packages. Please run: `pip install pymatgen mp-api`")

        if HAS_MP:
            st.success("✅ Materials Project libraries ready.")

            if st.button("🔍 Search Materials Project Database", type="primary", use_container_width=True):
                try:
                    # Accept commas or spaces between elements
                    elements = [e.strip() for e in search_query.replace(",", " ").split() if e.strip()]
                    with MPRester(mp_api_key) as mpr:
                        docs = mpr.materials.summary.search(
                            elements=elements,
                            fields=["material_id", "formula_pretty", "chemsys", "symmetry", "nsites", "density"],
                            chunk_size=max_cands
                        )
                    candidates = []
                    for doc in docs:
                        sym = getattr(doc, "symmetry", None)
                        candidates.append({
                            "id": doc.material_id,
                            "formula": getattr(doc, "formula_pretty", "N/A"),
                            "sg": getattr(sym, "symbol", "N/A") if sym else "N/A",
                            "crystal_system": getattr(sym, "crystal_system", "N/A") if sym else "N/A",
                            "nsites": getattr(doc, "nsites", "?"),
                            "density": round(getattr(doc, "density", 0), 2) if getattr(doc, "density", None) else None,
                            "doc": doc,  # keep original for later fetch
                            "source": "MP"
                        })
                    st.session_state["phase_candidates"] = candidates
                    st.session_state["phase_db"] = "MP"
                    st.session_state["mp_api_key"] = mp_api_key
                    st.success(f"Found **{len(candidates)}** candidate structures in Materials Project.")
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    if "API key" in str(e).lower() or "unauthorized" in str(e).lower():
                        st.info("Double-check your API key at https://materialsproject.org/")

# ========== COMMON CANDIDATE DISPLAY + MATCHING ==========
if "phase_candidates" in st.session_state and st.session_state["phase_candidates"]:
    candidates = st.session_state["phase_candidates"]
    db_used = st.session_state.get("phase_db", "Unknown")

    st.subheader(f"Candidates from {db_used} ({len(candidates)} found)")

    # --- Search / filter within the structure list ---
    filter_text = st.text_input(
        "🔍 Filter structures (type formula, elements, space group, or ID)",
        value="",
        placeholder="e.g. Fe2O3   or   Fe, O   or   R-3c   or   cod-901",
        help="Filters the list below in real time. Commas or spaces between elements both work (Fe, O or Fe O)."
    )

    def _matches_filter(c, query: str) -> bool:
        if not query or not query.strip():
            return True
        q = query.lower().strip()
        # Allow commas or spaces as separators
        tokens = [t.strip() for t in q.replace(",", " ").split() if t.strip()]
        searchable = " ".join([
            str(c.get("id", "")),
            str(c.get("formula", "")),
            str(c.get("sg", "")),
            str(c.get("crystal_system", "")),
        ]).lower()
        # Every token must appear somewhere
        return all(tok in searchable for tok in tokens)

    filtered = [c for c in candidates if _matches_filter(c, filter_text)]

    if filter_text.strip() and not filtered:
        st.warning(f"No structures match “{filter_text}”. Clear the filter or try different terms.")
    elif filter_text.strip():
        st.caption(f"Showing {len(filtered)} of {len(candidates)} structures")

    # Display filtered table
    cand_rows = []
    for c in filtered:
        row = {
            "ID": c["id"],
            "Formula": c["formula"],
            "Space Group": c.get("sg", "N/A"),
            "# Atoms": c.get("nsites", "?"),
        }
        if "crystal_system" in c:
            row["Crystal System"] = c["crystal_system"]
        if c.get("density") is not None:
            row["Density"] = c["density"]
        cand_rows.append(row)

    if cand_rows:
        st.dataframe(
            pd.DataFrame(cand_rows),
            use_container_width=True,
            hide_index=True,
            height=min(300, 45 + 32 * len(cand_rows))
        )
    else:
        st.info("No candidates to display.")

    # Selectbox only over the filtered list
    if filtered:
        selected_id = st.selectbox(
            "Choose a structure to simulate its XRD pattern and match against your peaks",
            options=[c["id"] for c in filtered],
            format_func=lambda x: f"{x} — {next((c['formula'] for c in filtered if c['id']==x), '')}"
        )
    else:
        selected_id = None

    tolerance = st.slider(
        "2θ matching tolerance (°)",
        min_value=0.05, max_value=0.5, value=0.15, step=0.05,
        help="How close a theoretical peak must be to an experimental one to count as a match."
    )

    if selected_id and st.button("📐 Simulate Pattern & Calculate Match Score", use_container_width=True):
        try:
            # Retrieve the Structure object (search in full list to be safe)
            selected = next(c for c in candidates if c["id"] == selected_id)

            if selected["source"] in ("COD", "AFLOW"):
                structure = selected["structure"]
            else:  # Materials Project
                with MPRester(st.session_state.get("mp_api_key", mp_api_key)) as mpr:
                    structure = mpr.get_structure_by_material_id(selected_id)

            # Simulate theoretical pattern
            xrd_calc = XRDCalculator(wavelength=wavelength)
            theo_pattern = xrd_calc.get_pattern(
                structure,
                two_theta_range=(float(x.min()), float(x.max()))
            )

            theo_2theta = np.array(theo_pattern.x)
            theo_intensity = np.array(theo_pattern.y)

            # Peak matching
            exp_2thetas = peaks_df["2θ (°)"].values if len(peaks_df) > 0 else np.array([])
            matches = []

            for exp_th in exp_2thetas:
                if len(theo_2theta) == 0:
                    continue
                diffs = np.abs(theo_2theta - exp_th)
                min_diff = diffs.min()
                if min_diff <= tolerance:
                    best_idx = np.argmin(diffs)
                    matches.append({
                        "Experimental 2θ": round(exp_th, 4),
                        "Theoretical 2θ": round(theo_2theta[best_idx], 4),
                        "Δ (°)": round(min_diff, 3),
                        "Theo. Intensity": round(theo_intensity[best_idx], 1),
                    })

            match_pct = (len(matches) / len(exp_2thetas) * 100) if len(exp_2thetas) > 0 else 0

            st.metric(
                "Peak Match Score",
                f"{match_pct:.1f}%",
                help=f"{len(matches)} out of {len(exp_2thetas)} experimental peaks matched within ±{tolerance}°"
            )

            if matches:
                st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
                st.caption("Higher match % + presence of strong theoretical peaks usually indicates a good candidate phase.")
            else:
                st.warning("No close matches found with current tolerance. Try increasing the tolerance or check if this phase is realistic for your sample.")

            # Crystal info
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(structure)
                crystal_sys = sga.get_crystal_system()
            except Exception:
                crystal_sys = "N/A"
            st.info(f"**{structure.composition.reduced_formula}** — {structure.get_space_group_info()[1]} ({crystal_sys})  ·  Source: {db_used}")

            # Persist for 3D + plane visualization
            st.session_state["last_structure"] = structure
            st.session_state["last_theo_pattern"] = theo_pattern
            st.session_state["last_source_id"] = selected_id

            # CIF download
            try:
                cif_content = structure.to(fmt="cif")
                st.download_button(
                    label="⬇️ Download CIF file",
                    data=cif_content,
                    file_name=f"{structure.composition.reduced_formula.replace(' ', '')}.cif",
                    mime="chemical/x-cif",
                    help="Download the crystal structure as a CIF file. You can open it in VESTA, Mercury, or upload it to the Crystal Structure Viewer below."
                )
            except Exception as cif_err:
                st.caption(f"Could not generate CIF: {cif_err}")

        except Exception as e:
            st.error(f"Failed to fetch/simulate structure: {e}")

# ========== 3D + (hkl) VISUALIZATION (shared) ==========
if "last_structure" in st.session_state:
    st.divider()
    st.subheader("🧊 3D Interactive Crystal Structure + Plane Visualization")

    struct = st.session_state["last_structure"]
    theo_pat = st.session_state.get("last_theo_pattern", None)

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Formula", struct.composition.reduced_formula)
    with col_info2:
        sg_info = struct.get_space_group_info()
        st.metric("Space Group", sg_info[1])
    with col_info3:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(struct)
            crystal_sys = sga.get_crystal_system()
        except Exception:
            crystal_sys = "N/A"
        st.metric("Crystal System", crystal_sys)

    lattice = struct.lattice
    st.caption(
        f"**Lattice parameters:** a = {lattice.a:.4f} Å | b = {lattice.b:.4f} Å | c = {lattice.c:.4f} Å  |  "
        f"α = {lattice.alpha:.1f}° β = {lattice.beta:.1f}° γ = {lattice.gamma:.1f}°  |  Volume = {lattice.volume:.2f} Å³"
    )

    show_3d = st.checkbox("🧊 Show interactive 3D crystal structure", value=True, key="show_3d_main")
    if show_3d:
        try:
            fig_3d = create_crystal_3d_plot(struct, title=f"{struct.composition.reduced_formula} - 3D View")
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
                st.caption("Interactive 3D view • Drag to rotate • Scroll to zoom • Different colors = different elements (see legend)")
        except Exception as viz_err:
            st.error(f"3D view error: {viz_err}")

    # Enhanced (hkl) matching table
    if theo_pat is not None and len(peaks_df) > 0:
        st.markdown("**Your experimental peaks matched to crystal planes (hkl) and d-spacings**")

        exp_2thetas = peaks_df["2θ (°)"].values
        theo_2theta = np.array(theo_pat.x)
        theo_intensity = np.array(theo_pat.y)
        theo_hkls = getattr(theo_pat, "hkls", [[]] * len(theo_2theta))

        enhanced = []
        tol = 0.15
        for exp_th in exp_2thetas:
            diffs = np.abs(theo_2theta - exp_th)
            if diffs.min() <= tol:
                idx = np.argmin(diffs)
                hkl_data = theo_hkls[idx] if idx < len(theo_hkls) else []
                hkl_str = str(hkl_data[0].get("hkl", "—")) if hkl_data and len(hkl_data) > 0 else "—"

                try:
                    d_val = lattice.d_hkl(hkl_data[0]["hkl"]) if hkl_data and len(hkl_data) > 0 else None
                except Exception:
                    d_val = peaks_df.loc[np.isclose(peaks_df["2θ (°)"], exp_th, atol=0.01), "d-spacing (Å)"].values
                    d_val = d_val[0] if len(d_val) > 0 else None

                enhanced.append({
                    "Exp 2θ (°)": round(exp_th, 4),
                    "Theo 2θ (°)": round(theo_2theta[idx], 4),
                    "Δ°": round(diffs.min(), 3),
                    "(hkl) plane": hkl_str,
                    "d-spacing (Å)": round(d_val, 4) if d_val else "—",
                    "Intensity (theo)": round(theo_intensity[idx], 1)
                })

        if enhanced:
            st.dataframe(pd.DataFrame(enhanced), use_container_width=True, hide_index=True)
            st.caption("Each matched peak corresponds to diffraction from a specific set of crystal planes (hkl). The d-spacing is the interplanar distance.")

            # Plane visualizer
            plane_options = []
            parsed_hkls = []
            for row in enhanced:
                hkl_str = row["(hkl) plane"]
                if hkl_str and hkl_str != "—":
                    try:
                        hkl = tuple(int(x) for x in hkl_str.strip("()[]").split(","))
                        if len(hkl) == 3:
                            label = f"({hkl[0]} {hkl[1]} {hkl[2]})  •  d = {row['d-spacing (Å)']} Å"
                            plane_options.append(label)
                            parsed_hkls.append(hkl)
                    except Exception:
                        continue

            if plane_options:
                st.markdown("**Visualize matched plane in 3D**")
                selected_label = st.selectbox(
                    "Choose a plane from your matched peaks:",
                    options=plane_options,
                    key="plane_selector"
                )

                if st.button("Show Selected Plane in 3D Viewer", key="show_selected_plane"):
                    try:
                        sel_idx = plane_options.index(selected_label)
                        h, k, l = parsed_hkls[sel_idx]

                        plane_points = get_unit_cell_plane_points(h, k, l, lattice)

                        if len(plane_points) >= 3:
                            fig_plane = create_crystal_3d_plot(
                                struct, title=f"{struct.composition.reduced_formula} + ({h} {k} {l}) plane"
                            )
                            x_p = [p[0] for p in plane_points]
                            y_p = [p[1] for p in plane_points]
                            z_p = [p[2] for p in plane_points]
                            fig_plane.add_trace(go.Mesh3d(
                                x=x_p, y=y_p, z=z_p,
                                color="rgba(255, 165, 0, 0.5)",
                                opacity=0.6,
                                name=f"({h} {k} {l})",
                                showlegend=True
                            ))
                            st.plotly_chart(fig_plane, use_container_width=True)
                            st.success(f"Showing the ({h} {k} {l}) plane corresponding to the selected matched peak.")
                        else:
                            st.warning("This plane does not intersect the unit cell well.")
                    except Exception as draw_err:
                        st.error(f"Could not draw the plane: {draw_err}")
        else:
            st.caption("No (hkl) details available at current tolerance.")

# ====================== STANDALONE CRYSTAL STRUCTURE VIEWER ======================
st.divider()
st.header("🧊 Crystal Structure Viewer")

st.markdown("""
Upload a `.cif` file to visualize any crystal structure in 3D.  
This works independently of the phase identification databases (COD / Materials Project).
""")

cif_file = st.file_uploader("Upload CIF file", type=["cif"], key="cif_viewer")

if cif_file is not None:
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        import tempfile
        import os

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
            tmp.write(cif_file.getvalue())
            tmp_path = tmp.name

        struct = Structure.from_file(tmp_path)
        os.unlink(tmp_path)  # clean up

        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Formula", struct.composition.reduced_formula)
        with col2:
            sg_info = struct.get_space_group_info()
            st.metric("Space Group", sg_info[1])
        with col3:
            try:
                sga = SpacegroupAnalyzer(struct)
                crystal_sys = sga.get_crystal_system()
            except:
                crystal_sys = "N/A"
            st.metric("Crystal System", crystal_sys)

        # Lattice
        lattice = struct.lattice
        st.caption(f"a = {lattice.a:.4f} Å, b = {lattice.b:.4f} Å, c = {lattice.c:.4f} Å | "
                   f"α = {lattice.alpha:.1f}°, β = {lattice.beta:.1f}°, γ = {lattice.gamma:.1f}° | "
                   f"Volume = {lattice.volume:.2f} Å³")

        # 3D Viewer (Plotly)
        if st.checkbox("Show 3D structure", value=True, key="standalone_3d"):
            try:
                fig_3d = create_crystal_3d_plot(struct, title=f"{struct.composition.reduced_formula} - Crystal Structure")
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
                    st.caption("Interactive 3D view • Colors represent different elements (see legend on the right)")
            except Exception as e:
                st.error(f"Failed to render 3D view: {e}")

        # Miller Plane Visualizer - helps users understand Miller indices visually
        st.markdown("### Visualize Specific Crystal Planes (hkl)")
        st.caption("This tool helps you visualize which crystal planes correspond to Miller indices. Enter h, k, l values and see the plane inside the unit cell (orange surface).")

        col_h, col_k, col_l = st.columns(3)
        with col_h:
            h_val = st.number_input("h", value=1, step=1, key="h_plane")
        with col_k:
            k_val = st.number_input("k", value=1, step=1, key="k_plane")
        with col_l:
            l_val = st.number_input("l", value=0, step=1, key="l_plane")

        if st.button("Show (hkl) Plane", key="show_plane_btn"):
            try:
                plane_points = get_unit_cell_plane_points(h_val, k_val, l_val, lattice)

                if len(plane_points) >= 3:
                    fig_with_plane = create_crystal_3d_plot(struct, title=f"{struct.composition.reduced_formula} + ({h_val}{k_val}{l_val}) plane")

                    x_p = [p[0] for p in plane_points]
                    y_p = [p[1] for p in plane_points]
                    z_p = [p[2] for p in plane_points]

                    fig_with_plane.add_trace(go.Mesh3d(
                        x=x_p, y=y_p, z=z_p,
                        color='rgba(255, 165, 0, 0.45)',
                        opacity=0.55,
                        name=f"({h_val} {k_val} {l_val})",
                        showlegend=True,
                        hovertemplate=f"({h_val} {k_val} {l_val}) plane<extra></extra>"
                    ))

                    st.plotly_chart(fig_with_plane, use_container_width=True)
                    st.success(f"Orange surface = the ({h_val} {k_val} {l_val}) crystal plane. This is the plane from which X-rays diffract at the corresponding 2θ angle.")
                else:
                    st.warning("This (hkl) combination does not produce a clear intersection with the unit cell. Try common values like (1,0,0), (1,1,0), or (1,1,1).")
            except Exception as plane_err:
                st.error(f"Could not draw plane: {plane_err}")

        # Optional: Show some low-index planes
        if st.checkbox("Show example (hkl) planes & d-spacings"):
            try:
                sga = SpacegroupAnalyzer(struct)
                # Get a few low index planes
                planes = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)]
                plane_data = []
                for hkl in planes:
                    try:
                        d = lattice.d_hkl(hkl)
                        plane_data.append({
                            "(hkl)": str(hkl),
                            "d-spacing (Å)": round(d, 4)
                        })
                    except:
                        pass
                if plane_data:
                    st.dataframe(pd.DataFrame(plane_data), use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"Could not calculate planes: {e}")

    except Exception as e:
        st.error(f"Failed to read CIF file: {e}")
        st.info("Make sure the file is a valid CIF format.")

st.caption("Built with ❤️ for researchers • Streamlit + pybaselines + Plotly + pymatgen + COD / AFLOW / Materials Project • Feedback welcome!")


# ====================== PEAK BROADENING ANALYSIS ======================
st.divider()
st.header("📏 Peak Broadening Analysis")

st.markdown("""
Analyze **crystallite size** and **microstrain** from peak broadening.

**Common sources of broadening:**
- Small crystallite size → Scherrer broadening
- Microstrain (lattice distortions) → Williamson-Hall analysis
- Instrumental broadening (subtracted first)
""")

if len(peaks) > 0:
    # Calculate FWHM from scipy find_peaks properties (in degrees 2θ)
    x_spacing = np.mean(np.diff(x))
    fwhm_deg = properties.get("widths", np.zeros(len(peaks))) * x_spacing

    # Create broadening dataframe
    broadening_data = []
    for i in range(len(peaks)):
        theta_deg = x[peaks[i]]
        theta_rad = np.deg2rad(theta_deg / 2)
        beta_rad = np.deg2rad(fwhm_deg[i])  # FWHM in radians

        broadening_data.append({
            "Peak #": i + 1,
            "2θ (°)": round(theta_deg, 4),
            "FWHM (°)": round(fwhm_deg[i], 4),
            "FWHM (rad)": round(beta_rad, 6),
            "cos θ": round(np.cos(theta_rad), 4),
            "sin θ": round(np.sin(theta_rad), 4),
        })

    broad_df = pd.DataFrame(broadening_data)
    st.dataframe(broad_df, use_container_width=True, hide_index=True)

    # === Scherrer Crystallite Size ===
    st.subheader("Scherrer Crystallite Size")

    col1, col2 = st.columns(2)
    with col1:
        K = st.number_input("Shape factor K", value=0.9, min_value=0.5, max_value=1.5, step=0.05,
                            help="0.9 for spherical crystallites (common default)")
    with col2:
        beta_inst = st.number_input("Instrumental FWHM (°) [optional]", value=0.0, min_value=0.0, step=0.01,
                                    help="Measure using a standard like LaB6 or Si. Set to 0 if unknown.")

    if st.button("Calculate Crystallite Sizes"):
        scherrer_results = []
        for i in range(len(peaks)):
            theta_deg = x[peaks[i]]
            theta_rad = np.deg2rad(theta_deg / 2)
            beta_sample = max(np.deg2rad(fwhm_deg[i]) - np.deg2rad(beta_inst), 1e-6)

            # Scherrer equation: D = K λ / (β cos θ)
            D_nm = (K * wavelength) / (beta_sample * np.cos(theta_rad)) / 10  # convert Å to nm

            scherrer_results.append({
                "Peak #": i + 1,
                "2θ (°)": round(theta_deg, 2),
                "FWHM corrected (°)": round(np.rad2deg(beta_sample), 4),
                "Crystallite Size (nm)": round(D_nm, 1),
            })

        sch_df = pd.DataFrame(scherrer_results)
        st.dataframe(sch_df, use_container_width=True, hide_index=True)

        avg_size = np.mean([r["Crystallite Size (nm)"] for r in scherrer_results])
        st.metric("Average Crystallite Size", f"{avg_size:.1f} nm")

    # === Williamson-Hall Plot (for microstrain) ===
    if len(peaks) >= 3:
        st.subheader("Williamson-Hall Analysis (Size + Strain)")

        st.caption("Plot β cos θ vs 4 sin θ. Slope = microstrain, intercept related to crystallite size.")

        if st.button("Generate Williamson-Hall Plot"):
            wh_x = []  # 4 sin θ
            wh_y = []  # β cos θ (in rad)

            for i in range(len(peaks)):
                theta_deg = x[peaks[i]]
                theta_rad = np.deg2rad(theta_deg / 2)
                beta_rad = np.deg2rad(fwhm_deg[i]) - np.deg2rad(beta_inst)
                beta_rad = max(beta_rad, 1e-6)

                wh_x.append(4 * np.sin(theta_rad))
                wh_y.append(beta_rad * np.cos(theta_rad))

            # Linear fit
            coeffs = np.polyfit(wh_x, wh_y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]

            # Microstrain ε = slope / 4
            microstrain = slope / 4
            # Size from intercept: D = K λ / intercept
            size_from_wh = (K * wavelength) / max(intercept, 1e-6) / 10  # nm

            # Create plot
            fig_wh = go.Figure()
            fig_wh.add_trace(go.Scatter(
                x=wh_x, y=wh_y,
                mode='markers+text',
                text=[f"({i+1})" for i in range(len(peaks))],
                textposition="top center",
                marker=dict(size=10, color="blue"),
                name="Data points"
            ))

            # Fitted line
            x_fit = np.linspace(min(wh_x), max(wh_x), 100)
            y_fit = np.polyval(coeffs, x_fit)
            fig_wh.add_trace(go.Scatter(
                x=x_fit, y=y_fit,
                mode='lines',
                line=dict(color="red", dash="dash"),
                name=f"Fit: ε = {microstrain:.4f}, D ≈ {size_from_wh:.1f} nm"
            ))

            fig_wh.update_layout(
                title="Williamson-Hall Plot",
                xaxis_title="4 sin θ",
                yaxis_title="β cos θ (rad)",
                height=450
            )
            st.plotly_chart(fig_wh, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Microstrain (ε)", f"{microstrain:.4f}")
            with col_b:
                st.metric("Crystallite Size (from WH)", f"{size_from_wh:.1f} nm")

            st.caption("Note: Williamson-Hall assumes isotropic strain and spherical crystallites. Results are approximate.")
else:
    st.info("Detect peaks first in the main analysis section to enable broadening analysis.")


# ====================== MILLER PLANE VISUALIZATION ======================
def get_unit_cell_plane_points(h, k, l, lattice):
    """
    Calculate the intersection points of the (hkl) plane with the unit cell.
    Returns a list of cartesian points forming the polygon.
    """
    if h == 0 and k == 0 and l == 0:
        return []

    # Plane equation in fractional coordinates: h*x + k*y + l*z = 1 (for non-zero)
    # We find intersections with the 12 edges of the unit cell [0,1]^3

    edges = [
        ((0,0,0), (1,0,0)), ((0,0,0), (0,1,0)), ((0,0,0), (0,0,1)),
        ((1,0,0), (1,1,0)), ((1,0,0), (1,0,1)),
        ((0,1,0), (1,1,0)), ((0,1,0), (0,1,1)),
        ((0,0,1), (1,0,1)), ((0,0,1), (0,1,1)),
        ((1,1,0), (1,1,1)), ((1,0,1), (1,1,1)), ((0,1,1), (1,1,1))
    ]

    points = []
    eps = 1e-8

    for p1, p2 in edges:
        # Parametric line: P = p1 + t*(p2 - p1), t in [0,1]
        dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]

        # Solve h*(x0 + t*dx) + k*(y0 + t*dy) + l*(z0 + t*dz) = 1
        denom = h*dx + k*dy + l*dz
        if abs(denom) < eps:
            continue

        t = (1 - (h*p1[0] + k*p1[1] + l*p1[2])) / denom

        if 0 <= t <= 1:
            x = p1[0] + t * dx
            y = p1[1] + t * dy
            z = p1[2] + t * dz
            cart = lattice.get_cartesian_coords([x, y, z])
            points.append(cart)

    # Remove duplicates and order them (simple convex hull approximation)
    if len(points) < 3:
        return []

    # Simple ordering around the plane normal
    normal = np.array([h, k, l], dtype=float)
    normal /= np.linalg.norm(normal)

    # Project to 2D for ordering
    if abs(normal[0]) > 0.5:
        basis1 = np.array([0, 1, 0])
    else:
        basis1 = np.array([1, 0, 0])
    basis2 = np.cross(normal, basis1)
    basis2 /= np.linalg.norm(basis2)

    projected = []
    for p in points:
        vec = np.array(p)
        u = np.dot(vec, basis1)
        v = np.dot(vec, basis2)
        projected.append((u, v, p))

    # Sort by angle
    center = np.mean([p[2] for p in projected], axis=0)
    projected.sort(key=lambda item: np.arctan2(
        np.dot(item[2] - center, basis2),
        np.dot(item[2] - center, basis1)
    ))

    return [p[2] for p in projected]