import os
import torch
import matplotlib.pyplot as plt

from models.encoder import Encoder
from models.decoder import Decoder
from data.synthetic_generator import PairedTimeSeriesGenerator

# ------------------------------------------------
# Config
# ------------------------------------------------

OBS_DIM = 2
LATENT_DIM = 2

# ------------------------------------------------
# Load data
# ------------------------------------------------

generator = PairedTimeSeriesGenerator(T=300)
data = generator.generate()

y1 = data["obs_1"].float()
y2 = data["obs_2"].float()

# ------------------------------------------------
# Load models
# ------------------------------------------------

encoder = Encoder(
    obs_dim=OBS_DIM,
    latent_dim=LATENT_DIM,
)

decoder = Decoder(
    obs_dim=OBS_DIM,
    latent_dim=LATENT_DIM,
)

encoder.load_state_dict(
    torch.load("checkpoints/encoder.pt", map_location="cpu")
)

decoder.load_state_dict(
    torch.load("checkpoints/decoder.pt", map_location="cpu")
)

encoder.eval()
decoder.eval()

# ------------------------------------------------
# Reconstruction
# ------------------------------------------------

with torch.no_grad():

    z, _ = encoder(y1, y2)

    y1_hat, y2_hat = decoder(z)

# ------------------------------------------------
# Plot
# ------------------------------------------------

os.makedirs("docs/figures", exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

# Observation 1

axes[0].plot(
    y1[:,0],
    label="Ground Truth",
    linewidth=2,
)

axes[0].plot(
    y1_hat[:,0],
    "--",
    linewidth=2,
    label="Prediction",
)

axes[0].set_title("Observation 1 Reconstruction")
axes[0].legend()

# Observation 2

axes[1].plot(
    y2[:,0],
    label="Ground Truth",
    linewidth=2,
)

axes[1].plot(
    y2_hat[:,0],
    "--",
    linewidth=2,
    label="Prediction",
)

axes[1].set_title("Observation 2 Reconstruction")
axes[1].legend()

plt.tight_layout()

plt.savefig(
    "docs/figures/reconstruction.png",
    dpi=300,
)

plt.show()

print("Saved docs/figures/reconstruction.png")