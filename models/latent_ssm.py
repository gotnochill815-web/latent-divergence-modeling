import torch
import torch.nn as nn
import torch.distributions as D


class LatentSSM(nn.Module):
    """
    Variational latent state-space model for paired time series.
    """

    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.joint_dim = 2 * latent_dim

        # State transition matrix (learnable)
        self.A = nn.Parameter(
            0.9 * torch.eye(self.joint_dim)
        )

        # Process noise (diagonal for stability)
        self.log_q = nn.Parameter(
            torch.zeros(self.joint_dim)
        )

    def transition(self, z_prev):
        """
        p(z_t | z_{t-1})
        """
        mean = z_prev @ self.A.T
        std = torch.exp(0.5 * self.log_q)

        return D.Normal(mean, std)

    def forward(self, z_prev):
        dist = self.transition(z_prev)
        z_t = dist.rsample()   # reparameterization trick
        return z_t, dist
