import torch

from data.synthetic_generator import PairedTimeSeriesGenerator
from experiments.train import train
from analysis.divergence_metrics import divergence_summary


def evaluate(
    T=300,
    epochs=150,
    lambda_div=5.0,
    beta_smooth=0.1,
):
    """
    Evaluate the latent divergence model.
    """

    print("=" * 60)
    print("Evaluating Latent Divergence Model")
    print("=" * 60)

    # -----------------------------
    # Generate dataset
    # -----------------------------
    generator = PairedTimeSeriesGenerator(T=T)
    data = generator.generate()

    # -----------------------------
    # Train model
    # -----------------------------
    encoder, _ = train(
        data,
        epochs=epochs,
        lambda_div=lambda_div,
        beta_smooth=beta_smooth,
    )

    encoder.eval()

    y1 = data["obs_1"]
    y2 = data["obs_2"]

    latent_dim = y1.shape[1]

    z_list = []

    with torch.no_grad():

        for t in range(T):

            z, _ = encoder(
                y1[t:t+1],
                y2[t:t+1]
            )

            z_list.append(z)

    z = torch.cat(z_list)

    z1 = z[:, :latent_dim]
    z2 = z[:, latent_dim:]

    metrics = divergence_summary(z1, z2)

    print()

    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")

    return metrics


if __name__ == "__main__":

    evaluate()
