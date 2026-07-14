import os
import torch

from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_ssm import LatentSSM

from data.synthetic_generator import PairedTimeSeriesGenerator
from analysis.divergence_metrics import divergence_summary

CHECKPOINT_DIR = "checkpoints"


class Evaluator:

    def __init__(
        self,
        T=300,
        device=None,
    ):

        self.T = T

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

    # ----------------------------------------------------
    # Load Trained Models
    # ----------------------------------------------------

    def load_models(self):

        obs_dim = 2
        latent_dim = 2

        encoder = Encoder(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
        ).to(self.device)

        decoder = Decoder(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
        ).to(self.device)

        ssm = LatentSSM(
            latent_dim=latent_dim,
        ).to(self.device)

        encoder.load_state_dict(
            torch.load(
                os.path.join(
                    CHECKPOINT_DIR,
                    "encoder.pt",
                ),
                map_location=self.device,
            )
        )

        decoder.load_state_dict(
            torch.load(
                os.path.join(
                    CHECKPOINT_DIR,
                    "decoder.pt",
                ),
                map_location=self.device,
            )
        )

        ssm.load_state_dict(
            torch.load(
                os.path.join(
                    CHECKPOINT_DIR,
                    "ssm.pt",
                ),
                map_location=self.device,
            )
        )

        encoder.eval()
        decoder.eval()
        ssm.eval()

        return encoder, decoder, ssm

    # ----------------------------------------------------
    # Evaluation
    # ----------------------------------------------------

    def run(self):

        print("=" * 70)
        print("Latent Divergence Evaluation")
        print("=" * 70)

        generator = PairedTimeSeriesGenerator(
            T=self.T
        )

        data = generator.generate()

        y1 = data["obs_1"].float().to(self.device)
        y2 = data["obs_2"].float().to(self.device)

        encoder, decoder, ssm = self.load_models()

        latent_dim = y1.shape[1]

        z_all = []

        recon1 = []
        recon2 = []

        with torch.no_grad():

            for t in range(self.T):

                z, _ = encoder(
                    y1[t:t+1],
                    y2[t:t+1],
                )

                y1_hat, y2_hat = decoder(z)

                recon1.append(y1_hat)
                recon2.append(y2_hat)

                z_all.append(z)

        z = torch.cat(z_all)

        recon1 = torch.cat(recon1)
        recon2 = torch.cat(recon2)

        z1 = z[:, :latent_dim]
        z2 = z[:, latent_dim:]

        metrics = divergence_summary(
            z1,
            z2,
        )

        print()

        print("Evaluation Metrics\n")

        for k, v in metrics.items():

            print(f"{k:20s}: {v:.4f}")

        return {

            "metrics": metrics,

            "z1": z1,

            "z2": z2,

            "y1": y1,

            "y2": y2,

            "y1_hat": recon1,

            "y2_hat": recon2,

        }


if __name__ == "__main__":

    evaluator = Evaluator()

    evaluator.run()