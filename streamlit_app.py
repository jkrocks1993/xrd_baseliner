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
    pip install streamlit pandas numpy scipy plotly pybaselines

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
                "arpls (Asymmetrically Reweighted PLS)",
                "polynomial",
                "modpoly (Modified Polynomial)",
                "imodpoly (Improved Modified Poly)",
                "rubberband",
                "None (no correction)"
            ],
            index=0,
            help="ALS and ARPLS are excellent for curved/sloping XRD backgrounds. Polynomial methods good for gentle curves."
        )
    else:
        st.warning("⚠️ pybaselines not installed. Using built-in ALS only.\n\nInstall with: `pip install pybaselines` for more methods.")
        baseline_method = st.selectbox(
            "Method",
            options=["als (built-in)", "linear detrend", "None"],
            index=0
        )
    
    # Method-specific parameters
    if "als" in baseline_method.lower():
        lam = st.number_input("λ (smoothness, higher = smoother baseline)", value=1e5, min_value=1e2, max_value=1e9, step=1e4, format="%.0e")
        p = st.slider("p (asymmetry, lower = more baseline below data)", min_value=0.001, max_value=0.5, value=0.01, step=0.001, format="%.3f")
        niter = st.slider("Iterations", min_value=5, max_value=30, value=10)
    elif "arpls" in baseline_method.lower() and HAS_PYBASELINES:
        lam = st.number_input("λ (smoothness)", value=1e5, min_value=1e2, max_value=1e9, step=1e4, format="%.0e")
    elif "poly" in baseline_method.lower() or "rubberband" in baseline_method.lower():
        poly_degree = st.slider("Polynomial degree", min_value=1, max_value=8, value=4)
    
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
        elif method_key == "arpls":
            baseline, _ = baseline_obj.arpls(y_proc, lam=lam)
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
    - **ALS / ARPLS**: Best for most XRD patterns with curved or sloping backgrounds. Increase λ for smoother baseline; decrease p if baseline is being pulled up into peaks.
    - **Polynomial / modpoly**: Good when background is gently curved. Higher degree = more flexible but can overfit.
    - **Rubberband**: Connects local minima — useful for very spiky data.
    
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

# ====================== PHASE IDENTIFICATION (Materials Project) ======================
st.header("🔬 Phase Identification")

with st.expander("ℹ️ Note on JCPDS / ICDD vs Free Alternatives", expanded=False):
    st.markdown("""
    **Official JCPDS/ICDD Powder Diffraction File (PDF)** is a **commercial paid database** — the gold standard but requires a license.

    **Excellent free alternative we support here**:
    - **Materials Project** (free API key at materialsproject.org)
    - Combined with `pymatgen` for realistic XRD pattern simulation from actual crystal structures.
    
    This approach is commonly used in research for inorganic materials phase identification. 
    You get crystal system, space group, lattice parameters, and a match score against your experimental peaks.
    """)

mp_api_key = st.text_input(
    "Materials Project API Key (get free at materialsproject.org)",
    type="password",
    placeholder="mp-XXXXXXXXXXXXXXXX",
    help="Your key is used only in the current browser session. It is never saved to disk or sent anywhere else."
)

if mp_api_key:
    try:
        from mp_api.client import MPRester
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        HAS_MP_LIBS = True
    except ImportError:
        HAS_MP_LIBS = False
        st.error("Missing packages. Please run:\n`pip install pymatgen mp-api`")
    
    if HAS_MP_LIBS:
        st.success("✅ Materials Project libraries loaded. Ready for phase search.")

        col1, col2 = st.columns([2, 1])
        with col1:
            elements_str = st.text_input(
                "Elements in your sample (comma-separated)",
                value="Fe, O",
                help="Example: Fe, O   or   Ca, C, O   or   Ti, O"
            )
        with col2:
            max_cands = st.slider("Max candidates", min_value=5, max_value=30, value=12, step=1)

        if st.button("🔍 Search Materials Project Database", type="primary", use_container_width=True):
            try:
                elements = [e.strip() for e in elements_str.split(",") if e.strip()]
                with MPRester(mp_api_key) as mpr:
                    docs = mpr.materials.summary.search(
                        elements=elements,
                        fields=[
                            "material_id", "formula_pretty", "chemsys",
                            "crystal_system", "spacegroup_symbol", "nsites", "density"
                        ],
                        chunk_size=max_cands
                    )
                st.session_state["mp_docs"] = docs
                st.success(f"Found **{len(docs)}** candidate structures in the {elements} chemical system(s).")
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                if "API key" in str(e).lower() or "unauthorized" in str(e).lower():
                    st.info("Double-check your API key at https://materialsproject.org/")

        if "mp_docs" in st.session_state:
            docs = st.session_state["mp_docs"]
            
            # Display candidates nicely
            cand_rows = []
            for doc in docs:
                cand_rows.append({
                    "ID": doc.material_id,
                    "Formula": getattr(doc, "formula_pretty", "N/A"),
                    "Crystal System": getattr(doc, "crystal_system", "N/A"),
                    "Space Group": getattr(doc, "spacegroup_symbol", "N/A"),
                    "# Atoms": getattr(doc, "nsites", "?"),
                    "Density": round(getattr(doc, "density", 0), 2) if getattr(doc, "density", None) else "—"
                })
            cand_df = pd.DataFrame(cand_rows)
            st.dataframe(cand_df, use_container_width=True, hide_index=True, height=220)

            selected_mp_id = st.selectbox(
                "Choose a structure to simulate its XRD pattern and match against your peaks",
                options=[d.material_id for d in docs],
                format_func=lambda x: f"{x} — {next((d.formula_pretty for d in docs if d.material_id==x), '')}"
            )

            tolerance = st.slider("2θ matching tolerance (°)", min_value=0.05, max_value=0.5, value=0.15, step=0.05,
                                  help="How close a theoretical peak must be to an experimental one to count as a match.")

            if st.button("📐 Simulate Pattern & Calculate Match Score", use_container_width=True):
                try:
                    with MPRester(mp_api_key) as mpr:
                        structure = mpr.get_structure_by_material_id(selected_mp_id)
                    
                    # Simulate theoretical pattern
                    xrd_calc = XRDCalculator(wavelength=wavelength)
                    theo_pattern = xrd_calc.get_pattern(
                        structure, 
                        two_theta_range=(float(x.min()), float(x.max()))
                    )
                    
                    theo_2theta = np.array(theo_pattern.x)
                    theo_intensity = np.array(theo_pattern.y)
                    
                    # Simple but effective peak matching
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
                    
                    st.metric("Peak Match Score", f"{match_pct:.1f}%", 
                              help=f"{len(matches)} out of {len(exp_2thetas)} experimental peaks matched within ±{tolerance}°")
                    
                    if matches:
                        st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
                        st.caption("Higher match % + presence of strong theoretical peaks usually indicates a good candidate phase.")
                    else:
                        st.warning("No close matches found with current tolerance. Try increasing the tolerance or check if this phase is realistic for your sample.")
                    
                    # Bonus: show some crystal info
                    st.info(f"**{structure.composition.reduced_formula}** — {structure.get_space_group_info()[1]} ({structure.get_crystal_system()})")

                except Exception as e:
                    st.error(f"Failed to fetch/simulate structure: {e}")
else:
    st.info("Enter a free Materials Project API key above to unlock automated phase identification against real crystal structures.")

st.caption("Built with ❤️ for researchers • Streamlit + pybaselines + Plotly + pymatgen (optional) • Feedback welcome!")
