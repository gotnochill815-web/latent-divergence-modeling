import torch
import torch.nn.functional as F
from torch.optim import Adam

from models.latent_ssm import LatentSSM
from models.encoder import Encoder
from losses.divergence_loss import divergence_loss, smoothness_loss


def train(
    data,
    latent_dim=2,
    obs_dim=2,
    epochs=200,
    lr=1e-3,
    lambda_div=1.0,
    beta_smooth=0.1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(obs_dim, latent_dim).to(device)
    ssm = LatentSSM(latent_dim).to(device)

    optimizer = Adam(
        list(encoder.parameters()) + list(ssm.parameters()),
        lr=lr
    )

    y1 = data["obs_1"].to(device)
    y2 = data["obs_2"].to(device)

    T = y1.shape[0]

    for epoch in range(epochs):
        optimizer.zero_grad()

        z_prev = torch.zeros(1, 2 * latent_dim).to(device)
        kl_total = 0.0
        z_seq = []

        for t in range(T):
            z_post, q_dist = encoder(
                y1[t:t+1],
                y2[t:t+1]
            )

            z_prior, p_dist = ssm(z_prev)

            kl = torch.distributions.kl_divergence(q_dist, p_dist).mean()
            kl_total += kl

            z_seq.append(z_post)
            z_prev = z_post.detach()

        z_seq = torch.cat(z_seq, dim=0)

        L_div = divergence_loss(z_seq, latent_dim)
        L_smooth = smoothness_loss(z_seq)

        loss = kl_total + lambda_div * L_div + beta_smooth * L_smooth
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"KL: {kl_total.item():.3f} | "
                f"Div: {L_div.item():.3f} | "
                f"Smooth: {L_smooth.item():.3f}"
            )

    return encoder, ssm
