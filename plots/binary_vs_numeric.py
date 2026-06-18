"""
binary_vs_numeric.py — Strip plots: binary feature on x, numeric on y, coloured by churn.
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "processed" / "cleaned_telco_customer_churn.csv"

df = pd.read_csv(DATA).dropna(subset=["total_charges"]).fillna(0)

COLORS = {0: "#4393c3", 1: "#d6604d"}
LABELS = {0: "No Churn", 1: "Churn"}
rng    = np.random.default_rng(42)


def strip(ax, binary_col: str, numeric_col: str,
          x_labels: tuple[str, str] = ("No", "Yes")) -> None:
    for churn_val in [0, 1]:
        for bin_val in [0, 1]:
            mask = (df["churn"] == churn_val) & (df[binary_col] == bin_val)
            n    = mask.sum()
            ax.scatter(
                bin_val + rng.uniform(-0.15, 0.15, n),
                df.loc[mask, numeric_col],
                color=COLORS[churn_val],
                alpha=0.2,
                s=6,
                label=LABELS[churn_val] if bin_val == 0 else "_nolegend_",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


NUMERIC_LABELS = {
    "tenure":          "Tenure (months)",
    "monthly_charges": "Monthly Charges ($)",
    "total_charges":   "Total Charges ($)",
}

# Each tuple: (binary_col, numeric_col, x_labels, title)
PLOTS = [
    ("contract_month-to-month",          "tenure",          ("No",          "Yes"),         "Month-to-Month Contract × Tenure"),
    ("contract_two_year",                "tenure",          ("No",          "Yes"),         "Two-Year Contract × Tenure"),
    ("internet_service_fiber_optic",     "monthly_charges", ("No",          "Yes"),         "Fiber Optic × Monthly Charges"),
    ("payment_method_electronic_check",  "monthly_charges", ("Other",       "E-Check"),     "Electronic Check × Monthly Charges"),
    ("online_security",                  "monthly_charges", ("No",          "Yes"),         "Online Security × Monthly Charges"),
    ("paperless_billing",                "tenure",          ("Paper Bill",  "Paperless"),   "Paperless Billing × Tenure"),
    ("partner",                          "tenure",          ("No Partner",  "Has Partner"), "Partner × Tenure"),
    ("dependents",                       "tenure",          ("No Deps.",    "Has Deps."),   "Dependents × Tenure"),
    ("senior_citizen",                   "total_charges",   ("Non-Senior",  "Senior"),      "Senior Citizen × Total Charges"),
    ("tech_support",                     "monthly_charges", ("No",          "Yes"),         "Tech Support × Monthly Charges"),
    ("streaming_tv",                     "tenure",          ("No",          "Yes"),         "Streaming TV × Tenure"),
    ("payment_method_mailed_check",      "tenure",          ("Other",       "Mailed Chk"), "Mailed Check × Tenure"),
]

n_plots = len(PLOTS)
ncols   = 3
nrows   = -(-n_plots // ncols)   # ceiling div

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8))
axes_flat  = axes.flatten()

for i, (bin_col, num_col, xlabels, title) in enumerate(PLOTS):
    ax = axes_flat[i]
    strip(ax, bin_col, num_col, xlabels)
    ax.set_title(title, fontsize=9.5, pad=5)
    ax.set_ylabel(NUMERIC_LABELS[num_col], fontsize=8)

    if i == 0:
        ax.legend(title="Churn", fontsize=7.5, title_fontsize=7.5,
                  markerscale=2, loc="upper right")

# hide unused axes
for j in range(n_plots, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle(
    "Binary Feature × Numeric Feature — coloured by Churn",
    fontsize=13, y=1.01,
)
fig.tight_layout()
out = HERE / "binary_vs_numeric.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
