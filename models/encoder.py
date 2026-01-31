import torch
import torch.nn as nn
import torch.distributions as D


class Encoder(nn.Module):
    """
    Amortized variational encoder:
    q(z_t | y_t^1, y_t^2)
    """

    def __init__(self, obs_dim, latent_dim, hidden_dim=64):
        super().__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.joint_dim = 2 * latent_dim

        self.net = nn.Sequential(
            nn.Linear(2 * obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu_head = nn.Linear(hidden_dim, self.joint_dim)
        self.logvar_head = nn.Linear(hidden_dim, self.joint_dim)

    def forward(self, y1, y2):
        """
        y1, y2: tensors of shape (B, obs_dim)
        """
        y = torch.cat([y1, y2], dim=-1)
        h = self.net(y)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        std = torch.exp(0.5 * logvar)
        dist = D.Normal(mu, std)

        z = dist.rsample()  # reparameterization
        return z, dist
