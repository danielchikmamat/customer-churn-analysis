import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "processed" / "cleaned_telco_customer_churn.csv"

df = pd.read_csv(DATA).dropna(subset=["total_charges"])

colors = {0: "#4393c3", 1: "#d6604d"}
labels = {0: "No Churn", 1: "Churn"}
x_labels = {0: "Non-Senior", 1: "Senior"}

fig, ax = plt.subplots(figsize=(8, 6))

rng = np.random.default_rng(42)

for churn_val in [0, 1]:
    for senior_val in [0, 1]:
        mask = (df["churn"] == churn_val) & (df["senior_citizen"] == senior_val)
        n = mask.sum()
        jitter = rng.uniform(-0.15, 0.15, n)
        ax.scatter(
            senior_val + jitter,
            df.loc[mask, "tenure"],
            color=colors[churn_val],
            alpha=0.25,
            s=8,
            label=labels[churn_val] if senior_val == 0 else "_nolegend_",
        )

ax.set_xticks([0, 1])
ax.set_xticklabels([x_labels[0], x_labels[1]], fontsize=12)
ax.set_xlabel("Senior Citizen", fontsize=12)
ax.set_ylabel("Tenure (months)", fontsize=12)
ax.set_title("Tenure by Senior Citizen Status\ncolored by Churn", fontsize=13)
ax.legend(title="Churn", fontsize=10, title_fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_xlim(-0.5, 1.5)

fig.tight_layout()
out = HERE / "senior_monthly_charges.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
