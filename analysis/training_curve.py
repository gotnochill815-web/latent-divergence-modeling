import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Load history
# -----------------------

history = pd.read_csv("results/training_history.csv")

os.makedirs("docs/figures", exist_ok=True)

# -----------------------
# Plot
# -----------------------

plt.figure(figsize=(12,6))

plt.plot(
    history["epoch"],
    history["total"],
    linewidth=3,
    label="Total Loss",
)

plt.plot(
    history["epoch"],
    history["reconstruction"],
    linewidth=2,
    label="Reconstruction",
)

plt.plot(
    history["epoch"],
    history["kl"],
    linewidth=2,
    label="KL",
)

plt.plot(
    history["epoch"],
    history["divergence"],
    linewidth=2,
    label="Divergence",
)

plt.plot(
    history["epoch"],
    history["smoothness"],
    linewidth=2,
    label="Smoothness",
)

plt.title(
    "Training Dynamics of the Latent Divergence Model",
    fontsize=16,
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

save_path = "docs/figures/training_curve.png"

plt.savefig(
    save_path,
    dpi=300,
)

print(f"Saved to {save_path}")

plt.show()
