import os
import sys
import re
import io

# Ensure gelphase package is importable on Streamlit Cloud (Linux)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st

from gelphase.deconvolution import fit_gel_lane
from gelphase.threshold import fit_saturation_threshold
from gelphase.visualizer import plot_spectral_decomposition, plot_threshold_fit

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GelPhase – Gel Profile Analyzer",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 GelPhase: Automated Gel Profile & Threshold Analyzer")
st.markdown(
    "Open-source tool for ImageJ gel-lane deconvolution (multi-Gaussian + AICc) "
    "and phase-transition saturation threshold identification (broken-line NLLS)."
)
st.info(
    "💡 **How to use:** Upload your ImageJ line-profile CSV files "
    "(2 columns: pixel, intensity). "
    "GelPhase auto-detects the FBS concentration from filenames like `...5%...`."
)

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📁 Upload ImageJ Lane Profile CSV files (select multiple):",
    type=["csv", "txt"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.write("👆 Upload CSV files above to begin.")
    st.stop()

st.success(f"✅ {len(uploaded_files)} file(s) selected.")

# ── Concentration confirmation ────────────────────────────────────────────────
st.markdown("### 1 · Confirm concentrations")
st.caption("Auto-detected from filenames — adjust if needed:")

file_concs = []
n_cols = min(4, len(uploaded_files))
cols = st.columns(n_cols)
for idx, f in enumerate(uploaded_files):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", f.name)
    detected = float(m.group(1)) if m else float(idx)
    with cols[idx % n_cols]:
        val = st.number_input(
            f.name[:22] + ("…" if len(f.name) > 22 else ""),
            value=detected, step=0.5,
            key=f"conc_{idx}",
        )
        file_concs.append((val, f))

st.markdown("---")

# ── Run analysis ──────────────────────────────────────────────────────────────
if not st.button("🚀 Run GelPhase Analysis", type="primary"):
    st.stop()

file_concs.sort(key=lambda t: t[0])
progress = st.progress(0)
status   = st.empty()

extracted_data   = []
spectral_figures = []

for i, (conc, fobj) in enumerate(file_concs):
    status.text(f"Processing {i+1}/{len(file_concs)}: {fobj.name} @ {conc}% …")
    fobj.seek(0)
    try:
        df = pd.read_csv(fobj, header=None)
        x  = df.iloc[:, 0].values.astype(float)
        y  = df.iloc[:, 1].values.astype(float)

        fit  = fit_gel_lane(x, y)
        fig  = plot_spectral_decomposition(fit, title=f"{fobj.name}  ({conc}%)")
        spectral_figures.append((conc, fig))

        extracted_data.append({
            "Filename":              fobj.name,
            "Concentration (%)":     conc,
            "Best Model (AICc)":     fit["best_model"],
            "AICc":                  round(fit["best_aic"], 1),
            "State-1 Centre (μ)":    round(fit["main_mu"],    2),
            "State-1 Width (σ)":     round(fit["main_sigma"], 2),
            "σ StdErr":              round(fit["main_sigma_err"], 2),
        })
    except Exception as e:
        st.error(f"⚠️ Error in {fobj.name}: {e}")
    progress.progress((i + 1) / len(file_concs))

status.empty()
progress.empty()

if not extracted_data:
    st.error("No files could be processed. Check your CSV format.")
    st.stop()

df_res = pd.DataFrame(extracted_data)

# ── Results table + CSV download ──────────────────────────────────────────────
st.markdown("### 2 · Extracted band parameters")
st.dataframe(df_res, use_container_width=True)

csv_buf = io.StringIO()
df_res.to_csv(csv_buf, index=False)
st.download_button(
    "📥 Download parameters CSV",
    data=csv_buf.getvalue(),
    file_name="gelphase_parameters.csv",
    mime="text/csv",
)

st.markdown("---")

# ── Threshold fit ─────────────────────────────────────────────────────────────
st.markdown("### 3 · Saturation threshold (piecewise broken-line NLLS)")

concs  = df_res["Concentration (%)"].values
sigmas = df_res["State-1 Width (σ)"].values
errs   = df_res["σ StdErr"].values

if len(concs) < 3:
    st.warning("⚠️ At least 3 concentration points are needed for threshold fitting.")
else:
    try:
        th  = fit_saturation_threshold(concs, sigmas, errs)
        fig_th = plot_threshold_fit(th, title="Saturation Threshold Curve")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🎯 Threshold  C_th", f"{th['threshold']:.2f} %")
        c2.metric("📉 Pre-slope  s₁",   f"{th['slope1']:.3f}")
        c3.metric("📈 Post-slope s₂",   f"{th['slope2']:.3f}")
        c4.metric("📊 Intercept  y₀",   f"{th['y_intercept']:.2f}")

        st.pyplot(fig_th)

        img_buf = io.BytesIO()
        fig_th.savefig(img_buf, format="png", dpi=300)
        st.download_button(
            "📥 Download threshold plot (PNG 300 dpi)",
            data=img_buf.getvalue(),
            file_name="gelphase_threshold.png",
            mime="image/png",
        )
    except Exception as e:
        st.error(f"Threshold fitting failed: {e}")

st.markdown("---")

# ── Individual spectral fits ──────────────────────────────────────────────────
st.markdown("### 4 · Individual band decomposition plots")
cols2 = st.columns(2)
for j, (conc, fig) in enumerate(spectral_figures):
    with cols2[j % 2]:
        st.pyplot(fig)
        plt.close(fig)

st.markdown("---")
st.caption("GelPhase v1.0.0 · BIOMOD Nanotechnology Research · MIT License · "
           "Developed with Google DeepMind Antigravity")
