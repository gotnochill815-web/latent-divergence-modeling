import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from data.synthetic_generator import PairedTimeSeriesGenerator

from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_ssm import LatentSSM

from analysis.divergence_metrics import divergence_summary


CHECKPOINT_DIR = "checkpoints"
RESULT_DIR = "results"


def load_models(obs_dim=2, latent_dim=2, device="cpu"):

    encoder = Encoder(obs_dim=obs_dim, latent_dim=latent_dim).to(device)
    decoder = Decoder(obs_dim=obs_dim, latent_dim=latent_dim).to(device)
    ssm = LatentSSM(latent_dim=latent_dim).to(device)

    encoder.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINT_DIR, "encoder.pt"),
            map_location=device,
        )
    )

    decoder.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINT_DIR, "decoder.pt"),
            map_location=device,
        )
    )

    ssm.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINT_DIR, "ssm.pt"),
            map_location=device,
        )
    )

    encoder.eval()
    decoder.eval()
    ssm.eval()

    return encoder, decoder, ssm


def visualize():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(RESULT_DIR, exist_ok=True)

    generator = PairedTimeSeriesGenerator(T=300)
    data = generator.generate()

    y1 = data["obs_1"].float().to(device)
    y2 = data["obs_2"].float().to(device)

    regime = data["regime"]

    obs_dim = y1.shape[1]
    latent_dim = obs_dim

    encoder, decoder, _ = load_models(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        device=device,
    )

    latent_list = []
    recon1 = []
    recon2 = []

    with torch.no_grad():

        for t in range(len(y1)):

            z, _ = encoder(
                y1[t:t+1],
                y2[t:t+1],
            )

            y1_hat, y2_hat = decoder(z)

            latent_list.append(z)

            recon1.append(y1_hat)
            recon2.append(y2_hat)

    z = torch.cat(latent_list)

    recon1 = torch.cat(recon1)
    recon2 = torch.cat(recon2)

    z1 = z[:, :latent_dim]
    z2 = z[:, latent_dim:]

    # ----------------------------------------------------
    # Smooth for visualization only
    # ----------------------------------------------------

    z1_plot = F.avg_pool1d(
        z1.T.unsqueeze(0),
        kernel_size=5,
        stride=1,
        padding=2,
    ).squeeze(0).T

    z2_plot = F.avg_pool1d(
        z2.T.unsqueeze(0),
        kernel_size=5,
        stride=1,
        padding=2,
    ).squeeze(0).T

    divergence = torch.norm(
        z1 - z2,
        dim=1,
    )

    metrics = divergence_summary(z1, z2)

    print("\nDivergence Metrics\n")

    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}")

    # =====================================================
    # Figure 1
    # =====================================================

    plt.figure(figsize=(10,4))

    plt.plot(
        z1_plot[:,0].cpu(),
        label="Agent 1",
    )

    plt.plot(
        z2_plot[:,0].cpu(),
        label="Agent 2",
    )

    plt.title("Latent Trajectories")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "latent_trajectory.png",
        ),
        dpi=300,
    )

    plt.show()

    # =====================================================
    # Figure 2
    # =====================================================

    plt.figure(figsize=(10,4))

    plt.plot(
        divergence.cpu(),
        linewidth=2,
        label="Latent Distance",
    )

    plt.axhline(
        divergence.mean().item(),
        linestyle="--",
        alpha=0.6,
        label="Mean",
    )

    plt.title("Latent Divergence")

    plt.xlabel("Time")

    plt.ylabel("Distance")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "latent_divergence.png",
        ),
        dpi=300,
    )

    plt.show()

    # =====================================================
    # Figure 3
    # =====================================================

    plt.figure(figsize=(10,4))

    plt.plot(
        divergence.cpu(),
        color="black",
        label="Distance",
    )

    for r in torch.unique(regime):

        idx = regime == r

        plt.scatter(
            torch.where(idx)[0],
            divergence[idx].cpu(),
            s=8,
            label=f"Regime {int(r)}",
        )

    plt.title("Regime Overlay")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "regime_overlay.png",
        ),
        dpi=300,
    )

    plt.show()

    # =====================================================
    # Figure 4
    # =====================================================

    plt.figure(figsize=(10,4))

    plt.plot(
        y1[:,0].cpu(),
        label="Original",
    )

    plt.plot(
        recon1[:,0].cpu(),
        label="Reconstructed",
    )

    plt.title("Observation Reconstruction")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "reconstruction.png",
        ),
        dpi=300,
    )

    plt.show()

    # =====================================================
    # Figure 5
    # =====================================================

    plt.figure(figsize=(6,6))

    plt.scatter(
        z1[:,0].cpu(),
        z1[:,1].cpu(),
        s=10,
        label="Agent 1",
    )

    plt.scatter(
        z2[:,0].cpu(),
        z2[:,1].cpu(),
        s=10,
        label="Agent 2",
    )

    plt.title("Latent Space")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "latent_space.png",
        ),
        dpi=300,
    )

    plt.show()


if __name__ == "__main__":

    visualize()