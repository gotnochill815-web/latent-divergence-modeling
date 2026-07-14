import os

import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from models.encoder import Encoder
from data.synthetic_generator import PairedTimeSeriesGenerator


# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="Latent Divergence Modeling",
    layout="wide",
)

# ==========================================================
# Title
# ==========================================================

st.title(" Latent Divergence Modeling")

st.markdown(
    """
Interactive research dashboard for **Variational State Space Models**
that learn divergence dynamics in paired non-stationary time series.
"""
)

st.divider()

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("Settings")

T = st.sidebar.slider(
    "Number of Time Steps",
    min_value=100,
    max_value=500,
    value=300,
    step=50,
)

run = st.sidebar.button(" Generate & Analyze")

# ==========================================================
# Load Encoder
# ==========================================================

@st.cache_resource
def load_encoder():

    encoder = Encoder(
        obs_dim=2,
        latent_dim=2,
    )

    encoder.load_state_dict(
        torch.load(
            "checkpoints/encoder.pt",
            map_location="cpu",
        )
    )

    encoder.eval()

    return encoder


encoder = load_encoder()

# ==========================================================
# Main Analysis
# ==========================================================

if run:

    with st.spinner("Running inference..."):

        generator = PairedTimeSeriesGenerator(
            T=T,
        )

        data = generator.generate()

        y1 = data["obs_1"]
        y2 = data["obs_2"]

        latent = []

        with torch.no_grad():

            for t in range(T):

                z, _ = encoder(
                    y1[t:t + 1],
                    y2[t:t + 1],
                )

                latent.append(z)

        latent = torch.cat(latent)

        z1 = latent[:, :2]
        z2 = latent[:, 2:]

        z1 = F.avg_pool1d(
            z1.T.unsqueeze(0),
            kernel_size=9,
            stride=1,
            padding=4,
        ).squeeze().T

        z2 = F.avg_pool1d(
            z2.T.unsqueeze(0),
            kernel_size=9,
            stride=1,
            padding=4,
        ).squeeze().T

        divergence = torch.norm(
            z1 - z2,
            dim=1,
        )

    # ======================================================
    # Metrics
    # ======================================================

    st.header(" Divergence Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Mean",
        f"{divergence.mean():.4f}",
    )

    c2.metric(
        "Maximum",
        f"{divergence.max():.4f}",
    )

    c3.metric(
        "Minimum",
        f"{divergence.min():.4f}",
    )

    c4.metric(
        "Std",
        f"{divergence.std():.4f}",
    )

    # ======================================================
    # Divergence Plot
    # ======================================================

    st.header(" Latent Divergence")

    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(
        divergence.numpy(),
        linewidth=3,
        label="Latent Divergence",
    )

    ax.axhline(
        divergence.mean(),
        linestyle="--",
        alpha=0.6,
        label="Mean",
    )

    ax.set_xlabel("Time")

    ax.set_ylabel("Distance")

    ax.grid(alpha=0.3)

    ax.legend()

    st.pyplot(fig)

    # ======================================================
    # Observations
    # ======================================================

    st.header(" Generated Time Series")

    fig, ax = plt.subplots(2,1,figsize=(12,6))

    ax[0].plot(y1[:,0])

    ax[0].set_title("Observation 1")

    ax[1].plot(y2[:,0])

    ax[1].set_title("Observation 2")

    plt.tight_layout()

    st.pyplot(fig)

else:

    st.info("Click **Generate & Analyze** from the sidebar.")

# ==========================================================
# Research Figures
# ==========================================================

st.divider()

st.header(" Research Figures")

figures = [

    (
        "Training Curve",
        "docs/figures/training_curve.png",
    ),

    (
        "Benchmark Summary",
        "docs/figures/benchmark_summary.png",
    ),

    (
        "Latent Space",
        "docs/figures/latent_space.png",
    ),

    (
        "Reconstruction",
        "docs/figures/reconstruction.png",
    ),

    (
        "Architecture",
        "docs/figures/architecture.png",
    ),

    (
        "Pipeline",
        "docs/figures/pipeline.png",
    ),

]

for title, path in figures:

    if os.path.exists(path):

        st.subheader(title)

        st.image(
            path,
            use_container_width=True,
        )

# ==========================================================
# Benchmark Results
# ==========================================================

if os.path.exists("results/benchmark_results.csv"):

    st.divider()

    st.header("🏆 Benchmark Results")

    df = pd.read_csv(
        "results/benchmark_results.csv",
    )

    st.dataframe(
        df,
        use_container_width=True,
    )

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.markdown(
    """
### About

This project demonstrates a **Variational State Space Model (VSSM)**
for learning divergence dynamics in paired time series.

Features:

- Variational Encoder
- Latent State Space Model
- Divergence-aware Loss
- Smoothness Regularization
- Benchmarking against Kalman, HMM and LSTM
- Interactive Visualization
"""
)
