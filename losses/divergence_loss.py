import torch


def divergence_loss(z, latent_dim):
    """
    Penalizes persistent separation between paired latent trajectories.

    z: Tensor (T, B, 2 * latent_dim)
    """
    z1 = z[..., :latent_dim]
    z2 = z[..., latent_dim:]

    dist = torch.norm(z1 - z2, dim=-1)   # (T, B)

    mean_dist = dist.mean(dim=0, keepdim=True)
    loss = ((dist - mean_dist) ** 2).mean()

    return loss


def smoothness_loss(z):
    """
    Temporal smoothness regularization.
    """
    return ((z[1:] - z[:-1]) ** 2).mean()
