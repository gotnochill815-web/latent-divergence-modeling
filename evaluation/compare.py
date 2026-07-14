import os
import pandas as pd
import matplotlib.pyplot as plt


RESULT_DIR = "results"

CSV_PATH = os.path.join(
    RESULT_DIR,
    "benchmark_results.csv",
)


class BenchmarkVisualizer:

    def __init__(self):

        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(
                f"{CSV_PATH} not found.\n"
                "Run benchmark.py first."
            )

        self.df = pd.read_csv(CSV_PATH)

    # --------------------------------------------------
    # Generic Bar Plot
    # --------------------------------------------------

    def plot_bar(
        self,
        column,
        title,
        ylabel,
        filename,
    ):

        if column not in self.df.columns:
            print(f"Skipping {column}")
            return

        plt.figure(figsize=(8,5))

        plt.bar(
            self.df["Model"],
            self.df[column],
            edgecolor="black",
        )

        plt.title(title)

        plt.ylabel(ylabel)

        plt.grid(
            axis="y",
            alpha=0.3,
        )

        plt.tight_layout()

        save_path = os.path.join(
            RESULT_DIR,
            filename,
        )

        plt.savefig(
            save_path,
            dpi=300,
        )

        plt.close()

        print(f"Saved {save_path}")

    # --------------------------------------------------
    # Summary Table
    # --------------------------------------------------

    def print_summary(self):

        print()

        print("=" * 70)

        print("Benchmark Summary")

        print("=" * 70)

        print(self.df)

        print()

    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    def run(self):

        self.print_summary()

        self.plot_bar(
            column="RMSE",
            title="RMSE Comparison",
            ylabel="RMSE",
            filename="rmse.png",
        )

        self.plot_bar(
            column="MAE",
            title="MAE Comparison",
            ylabel="MAE",
            filename="mae.png",
        )

        self.plot_bar(
            column="MSE",
            title="MSE Comparison",
            ylabel="MSE",
            filename="mse.png",
        )

        self.plot_bar(
            column="Runtime",
            title="Runtime Comparison",
            ylabel="Seconds",
            filename="runtime.png",
        )

        self.plot_bar(
            column="Mean Divergence",
            title="Mean Divergence",
            ylabel="Distance",
            filename="mean_divergence.png",
        )

        self.plot_bar(
            column="Max Divergence",
            title="Maximum Divergence",
            ylabel="Distance",
            filename="max_divergence.png",
        )

        self.plot_bar(
            column="Cosine Divergence",
            title="Cosine Divergence",
            ylabel="Distance",
            filename="cosine_divergence.png",
        )

        self.plot_bar(
            column="Smoothness",
            title="Latent Smoothness",
            ylabel="Loss",
            filename="smoothness.png",
        )

        print("\nComparison plots generated successfully!")


if __name__ == "__main__":

    BenchmarkVisualizer().run()