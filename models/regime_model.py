import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeInferenceModel(nn.Module):
    """
    Neural Regime Inference Network

    Infers discrete latent regimes from paired latent states.

    Example regimes:
        0 -> Stable
        1 -> Diverging
        2 -> Recovering
    """

    def __init__(
        self,
        latent_dim=4,
        hidden_dim=64,
        num_regimes=3,
        dropout=0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_regimes)
        )

    def forward(self, z1, z2):
        """
        Parameters
        ----------
        z1 : Tensor
            Shape (batch, latent_dim)

        z2 : Tensor
            Shape (batch, latent_dim)

        Returns
        -------
        logits : Tensor
            Shape (batch, num_regimes)

        probs : Tensor
            Shape (batch, num_regimes)

        prediction : Tensor
            Shape (batch,)
        """

        x = torch.cat([z1, z2], dim=-1)

        logits = self.network(x)

        probs = F.softmax(logits, dim=-1)

        prediction = torch.argmax(probs, dim=-1)

        return logits, probs, prediction
