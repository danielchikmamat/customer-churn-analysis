"""
Correlation matrix for cleaned Telco Customer Churn dataset (churn label excluded).

Method chosen per pair type:
  - Binary × binary   → phi coefficient  (contingency-table formula)
  - Binary × numeric  → Pearson R
  - Numeric × numeric → Pearson R
All methods use pairwise complete observations (NaN rows dropped per pair).
"""

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data" / "processed" / "cleaned_telco_customer_churn.csv"

df = pd.read_csv(DATA)

# ── Column classification ─────────────────────────────────────────────────────
binary_cols  = [c for c in df.columns if df[c].dropna().isin([0, 1]).all()]
numeric_cols = [c for c in df.columns if c not in binary_cols]


def phi(x: pd.Series, y: pd.Series) -> float:
    """Phi coefficient for two binary 0/1 series (pairwise complete obs)."""
    mask = x.notna() & y.notna()
    a, b = x[mask].astype(int), y[mask].astype(int)
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return float((n11 * n00 - n10 * n01) / denom) if denom else np.nan


def pearson(x: pd.Series, y: pd.Series) -> float:
    """Pearson R with pairwise complete observations."""
    mask = x.notna() & y.notna()
    a, b = x[mask], y[mask]
    if len(a) < 3:
        return np.nan
    r, _ = stats.pearsonr(a, b)
    return float(r)


# ── Build hybrid correlation matrix ──────────────────────────────────────────
cols = df.columns.tolist()
corr = pd.DataFrame(np.nan, index=cols, columns=cols)

for i, ci in enumerate(cols):
    corr.loc[ci, ci] = 1.0
    for j, cj in enumerate(cols):
        if j >= i:
            continue
        if ci in binary_cols and cj in binary_cols:
            r = phi(df[ci], df[cj])
        else:
            r = pearson(df[ci], df[cj])
        corr.loc[ci, cj] = r
        corr.loc[cj, ci] = r

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # lower triangle only

sns.heatmap(
    corr,
    mask=mask,
    ax=ax,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 6.5},
    linewidths=0.4,
    linecolor="white",
    square=True,
    cbar_kws={
        "shrink": 0.75,
        "label": "φ (binary×binary)  |  Pearson r (otherwise)",
    },
)

ax.set_title(
    "Correlation Matrix — Telco Customer Churn Features\n"
    "φ for binary×binary · Pearson r otherwise · churn included",
    fontsize=13,
    pad=14,
)
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)

plt.tight_layout()

out_path = HERE / "correlation_matrix.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close(fig)
