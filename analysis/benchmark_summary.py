import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------
# Load benchmark results
# ------------------------------------------

df = pd.read_csv("results/benchmark_results.csv")

os.makedirs("docs/figures", exist_ok=True)

# ------------------------------------------
# Plot
# ------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# RMSE
axes[0,0].bar(df["Model"], df["RMSE"])
axes[0,0].set_title("RMSE ↓")
axes[0,0].tick_params(axis='x', rotation=20)

# MAE
axes[0,1].bar(df["Model"], df["MAE"])
axes[0,1].set_title("MAE ↓")
axes[0,1].tick_params(axis='x', rotation=20)

# Runtime
axes[1,0].bar(df["Model"], df["Runtime"])
axes[1,0].set_title("Runtime (seconds)")
axes[1,0].tick_params(axis='x', rotation=20)

# Mean Divergence
if "Mean Divergence" in df.columns:

    div = df["Mean Divergence"].fillna(0)

else:

    div = [0] * len(df)

axes[1,1].bar(df["Model"], div)
axes[1,1].set_title("Mean Divergence")
axes[1,1].tick_params(axis='x', rotation=20)

plt.suptitle(
    "Benchmark Comparison",
    fontsize=18,
    weight="bold",
)

plt.tight_layout()

save_path = "docs/figures/benchmark_summary.png"

plt.savefig(
    save_path,
    dpi=300,
)

plt.show()

print(f"Saved {save_path}")