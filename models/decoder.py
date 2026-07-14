import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Observation decoder

    p(y_t | z_t)

    Maps the joint latent state back to paired observations.
    """

    def __init__(
        self,
        latent_dim=2,
        obs_dim=2,
        hidden_dim=64,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.joint_dim = 2 * latent_dim

        self.net = nn.Sequential(
            nn.Linear(self.joint_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 2 * obs_dim)
        )

    def forward(self, z):
        """
        Parameters
        ----------
        z : (B, 2*latent_dim)

        Returns
        -------
        y1_hat : (B, obs_dim)

        y2_hat : (B, obs_dim)
        """

        out = self.net(z)

        y1_hat = out[:, :self.obs_dim]
        y2_hat = out[:, self.obs_dim:]

        return y1_hat, y2_hat