import os
import torch
import torch.optim as optim
import pandas as pd

from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_ssm import LatentSSM

from losses.divergence_loss import (
    reconstruction_loss,
    kl_divergence,
    divergence_loss,
    smoothness_loss,
)


def train(
    data,
    epochs=150,
    lr=1e-3,
    beta=0.1,
    lambda_div=5.0,
    gamma=0.1,
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    y1 = data["obs_1"].float().to(device)
    y2 = data["obs_2"].float().to(device)

    obs_dim = y1.shape[1]
    latent_dim = obs_dim

    encoder = Encoder(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
    ).to(device)

    decoder = Decoder(
        obs_dim=obs_dim,
        latent_dim=latent_dim,
    ).to(device)

    ssm = LatentSSM(
        latent_dim=latent_dim,
    ).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(ssm.parameters()),
        lr=lr,
    )

    print("\nStarting Training...\n")

    history = {
        "epoch": [],
        "total": [],
        "reconstruction": [],
        "kl": [],
        "divergence": [],
        "smoothness": [],
    }

    for epoch in range(epochs):

        encoder.train()
        decoder.train()
        ssm.train()

        optimizer.zero_grad()

        latent_list = []
        posterior_list = []

        recon1 = []
        recon2 = []

        # ---------------------------------------
        # Encode entire sequence
        # ---------------------------------------

        for t in range(len(y1)):

            z, posterior = encoder(
                y1[t:t + 1],
                y2[t:t + 1],
            )

            latent_list.append(z)
            posterior_list.append(posterior)

            y1_hat, y2_hat = decoder(z)

            recon1.append(y1_hat)
            recon2.append(y2_hat)

        # ---------------------------------------
        # Stack sequence
        # ---------------------------------------

        z_sequence = torch.cat(latent_list, dim=0)

        recon1 = torch.cat(recon1, dim=0)
        recon2 = torch.cat(recon2, dim=0)

        z1 = z_sequence[:, :latent_dim]
        z2 = z_sequence[:, latent_dim:]

        # ---------------------------------------
        # Compute losses
        # ---------------------------------------

        recon = reconstruction_loss(
            y1,
            y2,
            recon1,
            recon2,
        )

        kl = 0.0

        for posterior in posterior_list:
            kl += kl_divergence(posterior)

        kl /= len(posterior_list)

        div = divergence_loss(
            z1,
            z2,
        )

        smooth = smoothness_loss(
            z_sequence,
        )

        total = (
            recon
            + beta * kl
            + lambda_div * div
            + gamma * smooth
        )

        total.backward()

        optimizer.step()

        # ---------------------------------------
        # Save history
        # ---------------------------------------

        history["epoch"].append(epoch)
        history["total"].append(total.item())
        history["reconstruction"].append(recon.item())
        history["kl"].append(kl.item())
        history["divergence"].append(div.item())
        history["smoothness"].append(smooth.item())

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d}"
                f" | Total {total.item():.4f}"
                f" | Recon {recon.item():.4f}"
                f" | KL {kl.item():.4f}"
                f" | Div {div.item():.4f}"
                f" | Smooth {smooth.item():.4f}"
            )

    print("\nTraining Finished.")

    # ---------------------------------------
    # Save checkpoints
    # ---------------------------------------

    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        encoder.state_dict(),
        "checkpoints/encoder.pt",
    )

    torch.save(
        decoder.state_dict(),
        "checkpoints/decoder.pt",
    )

    torch.save(
        ssm.state_dict(),
        "checkpoints/ssm.pt",
    )

    print("\nSaved checkpoints:")
    print("checkpoints/encoder.pt")
    print("checkpoints/decoder.pt")
    print("checkpoints/ssm.pt")

    # ---------------------------------------
    # Save training history
    # ---------------------------------------

    os.makedirs("results", exist_ok=True)

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        "results/training_history.csv",
        index=False,
    )

    print("\nSaved training history:")
    print("results/training_history.csv")

    return encoder, decoder, ssm


if __name__ == "__main__":

    from data.synthetic_generator import (
        PairedTimeSeriesGenerator,
    )

    generator = PairedTimeSeriesGenerator(
        T=300,
    )

    data = generator.generate()

    train(data)