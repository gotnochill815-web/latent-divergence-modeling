import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


# ============================================================
# Reconstruction Loss
# ============================================================

def reconstruction_loss(
    y1,
    y2,
    y1_hat,
    y2_hat,
):
    """
    Mean Squared Reconstruction Loss
    """

    loss1 = F.mse_loss(y1_hat, y1)
    loss2 = F.mse_loss(y2_hat, y2)

    return loss1 + loss2


# ============================================================
# KL Divergence
# ============================================================

def kl_divergence(
    posterior,
    prior=None,
):
    """
    KL(q(z|x) || p(z))

    If no prior is provided,
    assume standard Normal.
    """

    if prior is None:

        prior = D.Normal(
            torch.zeros_like(posterior.loc),
            torch.ones_like(posterior.scale),
        )

    kl = D.kl_divergence(
        posterior,
        prior,
    )

    return kl.mean()


# ============================================================
# Divergence Loss
# ============================================================

def divergence_loss(
    z1,
    z2,
):
    """
    Mean latent Euclidean distance.
    """

    return torch.norm(
        z1 - z2,
        dim=-1,
    ).mean()


# ============================================================
# Smoothness Loss
# ============================================================

def smoothness_loss(z_sequence):
    """
    Penalizes rapid changes in latent trajectories.
    """

    if z_sequence.shape[0] < 2:
        return torch.tensor(
            0.0,
            device=z_sequence.device,
        )

    dz = z_sequence[1:] - z_sequence[:-1]

    return (dz ** 2).mean()


# ============================================================
# Total Loss
# ============================================================

class TotalLoss(nn.Module):

    def __init__(
        self,
        beta=0.1,
        lambda_div=5.0,
        gamma=0.1,
    ):
        super().__init__()

        self.beta = beta
        self.lambda_div = lambda_div
        self.gamma = gamma

    def forward(
        self,
        y1,
        y2,
        y1_hat,
        y2_hat,
        posterior,
        z1,
        z2,
        z_sequence,
    ):

        recon = reconstruction_loss(
            y1,
            y2,
            y1_hat,
            y2_hat,
        )

        kl = kl_divergence(
            posterior,
        )

        div = divergence_loss(
            z1,
            z2,
        )

        smooth = smoothness_loss(
            z_sequence,
        )

        total = (
            recon
            + self.beta * kl
            + self.lambda_div * div
            + self.gamma * smooth
        )

        return total, {
            "reconstruction": recon.item(),
            "kl": kl.item(),
            "divergence": div.item(),
            "smoothness": smooth.item(),
        }