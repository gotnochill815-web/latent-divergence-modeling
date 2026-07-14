import torch

from data.synthetic_generator import PairedTimeSeriesGenerator
from experiments.train import train
from analysis.divergence_metrics import divergence_summary


class Evaluator:

    def __init__(
        self,
        T=300,
        epochs=150,
        lambda_div=5.0,
        beta_smooth=0.1,
    ):

        self.T = T
        self.epochs = epochs
        self.lambda_div = lambda_div
        self.beta_smooth = beta_smooth

    def run(self):

        print("=" * 70)
        print("Evaluating Latent Divergence Model")
        print("=" * 70)

        # -----------------------------
        # Generate Dataset
        # -----------------------------
        generator = PairedTimeSeriesGenerator(T=self.T)
        data = generator.generate()

        # -----------------------------
        # Train Model
        # -----------------------------
        encoder, ssm = train(
            data,
            epochs=self.epochs,
            lambda_div=self.lambda_div,
            beta_smooth=self.beta_smooth,
        )

        encoder.eval()

        y1 = data["obs_1"]
        y2 = data["obs_2"]

        latent_dim = y1.shape[1]

        z_list = []

        with torch.no_grad():

            for t in range(self.T):

                z, _ = encoder(
                    y1[t:t+1],
                    y2[t:t+1],
                )

                z_list.append(z)

        z = torch.cat(z_list)

        z1 = z[:, :latent_dim]
        z2 = z[:, latent_dim:]

        metrics = divergence_summary(
            z1,
            z2,
        )

        print()

        for key, value in metrics.items():
            print(f"{key:20s}: {value:.4f}")

        return metrics


if __name__ == "__main__":

    evaluator = Evaluator()

    evaluator.run()
