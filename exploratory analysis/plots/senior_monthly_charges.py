import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "processed" / "cleaned_telco_customer_churn.csv"

df = pd.read_csv(DATA).dropna(subset=["total_charges"])

colors   = {0: "#4393c3", 1: "#d6604d"}
labels   = {0: "No Churn", 1: "Churn"}
x_labels = {0: "Non-Senior", 1: "Senior"}

rng = np.random.default_rng(42)


def strip_plot(y_col: str, y_label: str, title: str, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for churn_val in [0, 1]:
        for senior_val in [0, 1]:
            mask = (df["churn"] == churn_val) & (df["senior_citizen"] == senior_val)
            n = mask.sum()
            jitter = rng.uniform(-0.15, 0.15, n)
            ax.scatter(
                senior_val + jitter,
                df.loc[mask, y_col],
                color=colors[churn_val],
                alpha=0.25,
                s=8,
                label=labels[churn_val] if senior_val == 0 else "_nolegend_",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([x_labels[0], x_labels[1]], fontsize=12)
    ax.set_xlabel("Senior Citizen", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(title="Churn", fontsize=10, title_fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xlim(-0.5, 1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


strip_plot(
    "tenure",
    "Tenure (months)",
    "Tenure by Senior Citizen Status\ncolored by Churn",
    HERE / "senior_tenure.png",
)

strip_plot(
    "monthly_charges",
    "Monthly Charges ($)",
    "Monthly Charges by Senior Citizen Status\ncolored by Churn",
    HERE / "senior_monthly_charges.png",
)
