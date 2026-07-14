import torch
from torch.utils.data import Dataset


class PairedTimeSeriesDataset(Dataset):
    """
    Dataset for paired time-series observations.

    Each sample consists of:
        Agent 1 observations
        Agent 2 observations
        Regime labels (optional)
    """

    def __init__(self, data):

        self.obs1 = data["obs_1"].float()
        self.obs2 = data["obs_2"].float()

        self.latent1 = data.get("latent_1", None)
        self.latent2 = data.get("latent_2", None)

        self.regime = data.get("regime", None)

    def __len__(self):
        return len(self.obs1)

    def __getitem__(self, idx):

        sample = {
            "obs1": self.obs1[idx],
            "obs2": self.obs2[idx],
        }

        if self.latent1 is not None:
            sample["latent1"] = self.latent1[idx]

        if self.latent2 is not None:
            sample["latent2"] = self.latent2[idx]

        if self.regime is not None:
            sample["regime"] = self.regime[idx]

        return sample
