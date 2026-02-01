import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Latent Divergence Modeling", layout="wide")

st.title("Latent Divergence Modeling")
st.write("App loaded successfully 🚀")


from models.encoder import Encoder
from data.synthetic_generator import PairedTimeSeriesGenerator

# --------------------------------------------------
# Page config (must be first Streamlit call)
# --------------------------------------------------
st.set_page_config(
    page_title="Latent Divergence Demo",
    layout="wide"
)

# --------------------------------------------------
# Static UI (always renders)
# --------------------------------------------------
st.title("Latent Divergence Modeling")
st.markdown(
    """
    Interactive **research demo** for latent divergence inference  
    in paired non-stationary time series.

    **How to use:**
    1. Choose number of time steps
    2. Click **Generate & Analyze**
    3. Observe divergence dynamics
    """
)

st.divider()

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Controls")
T = st.sidebar.slider("Time steps", 100, 500, 300, step=50)

# --------------------------------------------------
# Load model ONCE (cached)
# --------------------------------------------------
@st.cache_resource
def load_encoder():
    latent_dim = 2
    obs_dim = 2
    encoder = Encoder(obs_dim, latent_dim)
    encoder.load_state_dict(torch.load("checkpoints/encoder.pt", map_location="cpu"))
    encoder.eval()
    return encoder

encoder = load_encoder()
latent_dim = 2

# --------------------------------------------------
# Action button
# --------------------------------------------------
if st.button("Generate & Analyze"):
    with st.spinner("Running inference..."):
        # Generate data
        gen = PairedTimeSeriesGenerator(T=T)
        data = gen.generate()

        y1 = data["obs_1"]
        y2 = data["obs_2"]

        # Infer latents
        z_seq = []
        with torch.no_grad():
            for t in range(T):
                z, _ = encoder(
                    y1[t:t+1],
                    y2[t:t+1]
                )
                z_seq.append(z)

        z_seq = torch.cat(z_seq, dim=0)

        # Split latents
        z1 = z_seq[:, :latent_dim]
        z2 = z_seq[:, latent_dim:]

        # Smooth (visualization only)
        z1_s = F.avg_pool1d(
            z1.T.unsqueeze(0), kernel_size=9, stride=1, padding=4
        ).squeeze(0).T

        z2_s = F.avg_pool1d(
            z2.T.unsqueeze(0), kernel_size=9, stride=1, padding=4
        ).squeeze(0).T

        divergence = torch.norm(z1_s - z2_s, dim=1)

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(divergence, linewidth=2.5, label="Latent divergence")
        ax.axhline(
            divergence.mean().item(),
            linestyle="--",
            alpha=0.5,
            label="Mean divergence"
        )
        ax.set_title("Latent Divergence Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("‖x₁ − x₂‖")
        ax.grid(alpha=0.3)
        ax.legend()

        st.pyplot(fig)

else:
    st.info("Click **Generate & Analyze** to run the model.")
