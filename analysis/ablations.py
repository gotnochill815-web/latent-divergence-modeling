import pandas as pd

from analysis.evaluate import evaluate


ABLATIONS = [

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
        "name": "No Smoothness",
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

    print("\nRunning Ablation Studies\n")

    for config in ABLATIONS:

        print("=" * 70)
        print(config["name"])
        print("=" * 70)

        metrics = evaluate(
            lambda_div=config["lambda_div"],
            beta_smooth=config["beta_smooth"],
        )

        metrics["Model"] = config["name"]

        results.append(metrics)

    df = pd.DataFrame(results)

    cols = ["Model"] + [c for c in df.columns if c != "Model"]
    df = df[cols]

    print("\n")
    print(df)

    df.to_csv(
        "results/ablation_results.csv",
        index=False,
    )

    print("\nSaved results to results/ablation_results.csv")


if __name__ == "__main__":

    run_ablations()
