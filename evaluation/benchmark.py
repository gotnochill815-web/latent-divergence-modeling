import os
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data.synthetic_generator import PairedTimeSeriesGenerator

from baselines.kalman import KalmanBaseline
from baselines.hmm import HMMBaseline
from baselines.lstm import LSTMBaseline

from evaluation.metrics import evaluate_model
from evaluation.interface import VSSMInterface


RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


class Benchmark:

    def __init__(self, T=300):

        self.T = T

    def generate_dataset(self):

        generator = PairedTimeSeriesGenerator(T=self.T)

        data = generator.generate()

        return data

    # ---------------------------------------------------
    # Kalman
    # ---------------------------------------------------

    def evaluate_kalman(self, data):

        observations = data["obs_1"].numpy()

        model = KalmanBaseline(dim=2)

        pred, runtime = model.run(observations)

        return evaluate_model(
            model_name="Kalman",
            y_true=torch.tensor(observations).float(),
            y_pred=torch.tensor(pred).float(),
            runtime=runtime,
        )

    # ---------------------------------------------------
    # HMM
    # ---------------------------------------------------

    def evaluate_hmm(self, data):

        observations = data["obs_1"].numpy()

        model = HMMBaseline()

        model.fit(observations)

        pred, states, runtime = model.run(observations)

        return evaluate_model(
            model_name="HMM",
            y_true=torch.tensor(observations).float(),
            y_pred=torch.tensor(pred).float(),
            runtime=runtime,
        )

    # ---------------------------------------------------
    # LSTM
    # ---------------------------------------------------

    def evaluate_lstm(self, data):

        y1 = data["obs_1"]

        y2 = data["obs_2"]

        x = torch.cat([y1, y2], dim=1)

        model = LSTMBaseline()

        pred, runtime = model.run(x)

        pred = pred[:, :2]

        return evaluate_model(
            model_name="LSTM",
            y_true=y1,
            y_pred=pred,
            runtime=runtime,
        )

    # ---------------------------------------------------
    # Our Model
    # ---------------------------------------------------

    def evaluate_vssm(self):

        model = VSSMInterface()

        result = model.run()

        return evaluate_model(
            model_name="VSSM (Ours)",
            y_true=result["y_true"].cpu(),
            y_pred=result["y_pred"].cpu(),
            z1=result["z1"].cpu(),
            z2=result["z2"].cpu(),
            runtime=result["runtime"],
        )

    # ---------------------------------------------------
    # Plot
    # ---------------------------------------------------

    def plot_metric(self, df, metric, filename):

        plt.figure(figsize=(8,4))

        plt.bar(df["Model"], df[metric])

        plt.title(metric)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULT_DIR,
                filename,
            ),
            dpi=300,
        )

        plt.close()

    # ---------------------------------------------------
    # Run
    # ---------------------------------------------------

    def run(self):

        print("="*60)
        print("Running Benchmark")
        print("="*60)

        data = self.generate_dataset()

        results = []

        results.append(
            self.evaluate_kalman(data)
        )

        results.append(
            self.evaluate_hmm(data)
        )

        results.append(
            self.evaluate_lstm(data)
        )

        results.append(
            self.evaluate_vssm()
        )

        df = pd.DataFrame(results)

        csv_path = os.path.join(
            RESULT_DIR,
            "benchmark_results.csv",
        )

        df.to_csv(csv_path, index=False)

        print(df)

        self.plot_metric(
            df,
            "RMSE",
            "benchmark_rmse.png",
        )

        self.plot_metric(
            df,
            "Runtime",
            "benchmark_runtime.png",
        )

        if "Mean Divergence" in df.columns:

            self.plot_metric(
                df,
                "Mean Divergence",
                "benchmark_divergence.png",
            )

        print()

        print("Saved Results")

        print(csv_path)


if __name__ == "__main__":

    benchmark = Benchmark()

    benchmark.run()