"""
mutual_information.py — Mutual information between each feature and churn.

Uses sklearn's mutual_info_classif which handles both continuous and
discrete features correctly:
  - Binary / discrete columns : treated as discrete (discrete_features=True)
  - Numeric columns           : treated as continuous (k-NN estimator)

Produces:
  mi_scores.csv          — raw MI scores ranked highest to lowest
  mi_bar_chart.png       — horizontal bar chart of all features
  mi_top10.png           — zoomed bar chart of top-10 features

Dependencies: pandas, numpy, scikit-learn, matplotlib
Run:          python mutual_information.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

HERE     = pathlib.Path(__file__).resolve().parent
DATA     = HERE.parent / "data" / "processed" / "cleaned_telco_customer_churn.csv"
OUT_DIR  = HERE
RAND     = 42

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA)
df = df.dropna(subset=["total_charges"])
df = df.fillna(0)

y = df["churn"].astype(int)
X = df.drop(columns=["churn"])

# ── Classify columns as discrete or continuous ─────────────────────────────────

binary_cols  = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
numeric_cols = [c for c in X.columns if c not in binary_cols]

discrete_mask = np.array([c in binary_cols for c in X.columns])

print(f"Features  : {X.shape[1]}")
print(f"Discrete  : {discrete_mask.sum()}  → {binary_cols}")
print(f"Continuous: {(~discrete_mask).sum()}  → {numeric_cols}")

# ── Compute MI ────────────────────────────────────────────────────────────────

mi_scores = mutual_info_classif(
    X.values.astype(float),
    y,
    discrete_features=discrete_mask,
    random_state=RAND,
)

mi_series = (
    pd.Series(mi_scores, index=X.columns, name="mutual_information")
    .sort_values(ascending=False)
)

print(f"\nMutual Information vs churn (bits)\n{'─'*45}")
print(mi_series.to_string())

# ── Save CSV ──────────────────────────────────────────────────────────────────

csv_path = OUT_DIR / "mi_scores.csv"
mi_series.reset_index().rename(columns={"index": "feature"}).to_csv(csv_path, index=False)
print(f"\nSaved → {csv_path}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar_chart(scores: pd.Series, title: str, out_path: pathlib.Path) -> None:
    n     = len(scores)
    colors = ["#d73027" if v >= scores.quantile(0.75) else
              "#fc8d59" if v >= scores.median()       else
              "#91bfdb" for v in scores.values]

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.38)))
    bars = ax.barh(scores.index[::-1], scores.values[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.4)

    # value labels
    for bar, val in zip(bars, scores.values[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.set_xlabel("Mutual Information (bits)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlim(0, scores.max() * 1.15)
    ax.axvline(scores.median(), color="gray", linestyle="--", linewidth=0.8,
               label=f"Median ({scores.median():.3f})")
    ax.legend(fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Full bar chart ────────────────────────────────────────────────────────────

_bar_chart(
    mi_series,
    "Mutual Information — All Features vs Churn",
    OUT_DIR / "mi_bar_chart.png",
)

# ── Top-10 bar chart ──────────────────────────────────────────────────────────

_bar_chart(
    mi_series.head(10),
    "Mutual Information — Top 10 Features vs Churn",
    OUT_DIR / "mi_top10.png",
)
