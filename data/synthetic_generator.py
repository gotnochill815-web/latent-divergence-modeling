import numpy as np
import torch


class PairedTimeSeriesGenerator:
    """
    Synthetic generator for paired non-stationary time series with
    latent coupling, divergence, and regime shifts.
    """

    def __init__(
        self,
        T=500,
        latent_dim=2,
        obs_dim=2,
        noise_std=0.1,
        seed=42
    ):
        self.T = T
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.noise_std = noise_std

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Linear latent dynamics (stable)
        self.A = 0.95 * np.eye(latent_dim)

        # Observation matrix
        self.C = np.random.randn(obs_dim, latent_dim)

    def _regime_schedule(self):
        """
        Defines regime-dependent coupling strength gamma_t.
        """
        gamma = np.zeros(self.T)
        regime = np.zeros(self.T, dtype=int)

        for t in range(self.T):
            if t < 150:
                gamma[t] = 0.8        # stable
                regime[t] = 0
            elif t < 250:
                gamma[t] = 0.1        # drift
                regime[t] = 1
            elif t < 350:
                gamma[t] = 0.0        # shock
                regime[t] = 2
            else:
                gamma[t] = 0.5        # recovery
                regime[t] = 3

        return gamma, regime

    def generate(self):
        """
        Returns paired observations with ground-truth latent states.
        """

        x1 = np.zeros((self.T, self.latent_dim))
        x2 = np.zeros((self.T, self.latent_dim))

        # initial latent states
        x1[0] = np.random.randn(self.latent_dim)
        x2[0] = np.random.randn(self.latent_dim)

        gamma, regime = self._regime_schedule()

        for t in range(1, self.T):
            noise1 = np.random.randn(self.latent_dim) * self.noise_std
            noise2 = np.random.randn(self.latent_dim) * self.noise_std

            coupling_1 = gamma[t] * (x2[t-1] - x1[t-1])
            coupling_2 = gamma[t] * (x1[t-1] - x2[t-1])

            x1[t] = self.A @ x1[t-1] + coupling_1 + noise1
            x2[t] = self.A @ x2[t-1] + coupling_2 + noise2

        # Nonlinear observation model
        y1 = np.tanh(x1 @ self.C.T) + np.random.randn(self.T, self.obs_dim) * self.noise_std
        y2 = np.tanh(x2 @ self.C.T) + np.random.randn(self.T, self.obs_dim) * self.noise_std

        return {
            "obs_1": torch.tensor(y1, dtype=torch.float32),
            "obs_2": torch.tensor(y2, dtype=torch.float32),
            "latent_1": torch.tensor(x1, dtype=torch.float32),
            "latent_2": torch.tensor(x2, dtype=torch.float32),
            "regime": torch.tensor(regime, dtype=torch.long),
            "gamma": torch.tensor(gamma, dtype=torch.float32)
        }
