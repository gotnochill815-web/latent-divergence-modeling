import pandas as pd

from experiments.evaluate import Evaluator


ABLATION_CONFIGS = [

    {
        "name": "Full Model",
        "lambda_div": 5.0,
        "beta_smooth": 0.1,
    },

    {
        "name": "No Divergence Loss",
        "lambda_div": 0.0,
        "beta_smooth": 0.1,
    },

    {
        "name": "No Smoothness Loss",
        "lambda_div": 5.0,
        "beta_smooth": 0.0,
    },

    {
        "name": "Weak Divergence",
        "lambda_div": 1.0,
        "beta_smooth": 0.1,
    },

]


def run_ablations():

    results = []

    print("\nRunning Ablation Experiments\n")

    for config in ABLATION_CONFIGS:

        print("=" * 70)
        print(config["name"])
        print("=" * 70)

        evaluator = Evaluator(
            lambda_div=config["lambda_div"],
            beta_smooth=config["beta_smooth"],
        )

        metrics = evaluator.run()

        metrics["Model"] = config["name"]

        results.append(metrics)

    df = pd.DataFrame(results)

    columns = ["Model"] + [
        c for c in df.columns if c != "Model"
    ]

    df = df[columns]

    print("\n")
    print(df)

    df.to_csv(
        "results/ablation_results.csv",
        index=False,
    )

    print("\nResults saved to:")
    print("results/ablation_results.csv")


if __name__ == "__main__":

    run_ablations()
