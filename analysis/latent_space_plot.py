import os
import torch
import matplotlib.pyplot as plt

from models.encoder import Encoder
from data.synthetic_generator import PairedTimeSeriesGenerator

# ---------------------------------------------
# Configuration
# ---------------------------------------------

OBS_DIM = 2
LATENT_DIM = 2

# ---------------------------------------------
# Generate data
# ---------------------------------------------

generator = PairedTimeSeriesGenerator(T=300)

data = generator.generate()

y1 = data["obs_1"].float()
y2 = data["obs_2"].float()

# ---------------------------------------------
# Load encoder
# ---------------------------------------------

encoder = Encoder(
    obs_dim=OBS_DIM,
    latent_dim=LATENT_DIM,
)

encoder.load_state_dict(
    torch.load(
        "checkpoints/encoder.pt",
        map_location="cpu",
    )
)

encoder.eval()

# ---------------------------------------------
# Encode
# ---------------------------------------------

with torch.no_grad():

    z, _ = encoder(
        y1,
        y2,
    )

z1 = z[:, :LATENT_DIM]
z2 = z[:, LATENT_DIM:]

# ---------------------------------------------
# Plot
# ---------------------------------------------

os.makedirs("docs/figures", exist_ok=True)

plt.figure(figsize=(8,8))

plt.plot(
    z1[:,0],
    z1[:,1],
    linewidth=2,
    label="Latent Stream 1",
)

plt.plot(
    z2[:,0],
    z2[:,1],
    linewidth=2,
    label="Latent Stream 2",
)

plt.scatter(
    z1[0,0],
    z1[0,1],
    s=80,
    marker="o",
    label="Start",
)

plt.scatter(
    z1[-1,0],
    z1[-1,1],
    s=80,
    marker="X",
    label="End",
)

plt.xlabel("Latent Dimension 1")

plt.ylabel("Latent Dimension 2")

plt.title(
    "Latent Trajectories",
    fontsize=16,
)

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "docs/figures/latent_space.png",
    dpi=300,
)

plt.show()

print("Saved docs/figures/latent_space.png")