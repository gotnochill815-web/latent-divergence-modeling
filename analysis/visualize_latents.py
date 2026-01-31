import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from data.synthetic_generator import PairedTimeSeriesGenerator
from experiments.train import train


def visualize():
    # -----------------------------
    # Generate synthetic data
    # -----------------------------
    gen = PairedTimeSeriesGenerator(T=300)
    data = gen.generate()

    # -----------------------------
    # Train model
    # -----------------------------
    encoder, _ = train(
        data,
        epochs=150,
        lambda_div=3.0,     # strong divergence signal
        beta_smooth=0.1
    )

    encoder.eval()

    y1 = data["obs_1"]
    y2 = data["obs_2"]

    latent_dim = y1.shape[1]
    z_seq = []

    # -----------------------------
    # Infer latents
    # -----------------------------
    with torch.no_grad():
        for t in range(y1.shape[0]):
            z, _ = encoder(
                y1[t:t+1],
                y2[t:t+1]
            )
            z_seq.append(z)

    z_seq = torch.cat(z_seq, dim=0)

    # -----------------------------
    # Split latents
    # -----------------------------
    z1 = z_seq[:, :latent_dim]
    z2 = z_seq[:, latent_dim:]

    # -----------------------------
    # Smooth latents (visualization only)
    # -----------------------------
    z1_smooth = F.avg_pool1d(
        z1.T.unsqueeze(0),
        kernel_size=9,
        stride=1,
        padding=4
    ).squeeze(0).T

    z2_smooth = F.avg_pool1d(
        z2.T.unsqueeze(0),
        kernel_size=9,
        stride=1,
        padding=4
    ).squeeze(0).T

    # -----------------------------
    # FINAL FIGURE: Divergence
    # -----------------------------
    divergence = torch.norm(z1_smooth - z2_smooth, dim=1)

    plt.figure(figsize=(12, 4))
    plt.plot(divergence, linewidth=2.5)
    plt.title("Latent Divergence Between Paired Agents", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("‖x₁ − x₂‖", fontsize=12)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/latent_divergence.png", dpi=300)
    plt.savefig("results/latent_divergence.pdf")
    plt.show()


if __name__ == "__main__":
    visualize()
