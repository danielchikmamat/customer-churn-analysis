"""
kmeans_elbow.py — K-Means clustering sweep with elbow + silhouette analysis
            on the cleaned Telco customer churn dataset.

Dependencies
------------
    numpy, pandas, scikit-learn, matplotlib
    kneed  (optional — pip install kneed; falls back to max-chord-distance)

Usage
-----
    python kmeans_elbow.py [--k_max INT] [--subsample INT] [--data PATH]

    --k_max      Maximum k to test            (default: 10)
    --subsample  Max rows for silhouette score (default: 2000)
    --data       Path to cleaned CSV          (default: data/processed/cleaned_telco_customer_churn.csv)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # safe for headless; swap to TkAgg/Qt5Agg for interactive windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from kneed import KneeLocator
    _KNEED = True
except ImportError:
    _KNEED = False

# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent   # project root (one level above kmeans/)
DEFAULT_DATA = ROOT / "data" / "processed" / "cleaned_telco_customer_churn.csv"
PLOTS_DIR   = ROOT / "kmeans plots"
PLOTS_DIR.mkdir(exist_ok=True)
# ---------------------------------------------------------------------------


# ── 1. Load ─────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    """Read the cleaned churn CSV from *path* and return a DataFrame."""
    print(f"\n[load_data] Loading from: {path.resolve()}")
    return pd.read_csv(path)


# ── 2. Inspect ──────────────────────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> None:
    """Print shape, column names, dtypes, head, and missing-value counts."""
    print(f"\n[inspect_data] Shape   : {df.shape}")
    print(f"[inspect_data] Columns : {df.columns.tolist()}")
    print("\n[inspect_data] dtypes:")
    print(df.dtypes.to_string())
    print("\n[inspect_data] head(3):")
    print(df.head(3).to_string())
    missing = df.isnull().sum()
    if missing.any():
        print("\n[inspect_data] Missing values (non-zero only):")
        print(missing[missing > 0].to_string())
    else:
        print("\n[inspect_data] No missing values detected.")


# ── 3. Preprocess ───────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, list[str]]:
    """
    Prepare a scaled feature matrix for clustering.

    Steps
    -----
    1. Separate and drop the *churn* target column.
    2. Coerce all remaining columns to numeric (catches stray strings).
    3. Fill residual NaN with 0 — all NaN cells are binary service columns
       where NaN means "no applicable service", semantically equivalent to 0.
    4. StandardScaler-normalise every feature.

    Returns
    -------
    X_scaled      : (n_samples, n_features) float64 array
    churn         : original 0/1 Series for optional post-hoc comparison
    feature_names : list of column names
    """
    if "churn" not in df.columns:
        raise ValueError("Expected a 'churn' column — not found.")

    churn = df["churn"].copy()
    X = df.drop(columns=["churn"])

    print(f"\n[preprocess] {len(X.columns)} feature columns:")
    print(X.columns.tolist())

    # coerce — catches cases like TotalCharges loaded as str
    X = X.apply(pd.to_numeric, errors="coerce")

    # drop rows where total_charges is NaN before any filling
    if "total_charges" in X.columns:
        n_drop = X["total_charges"].isnull().sum()
        if n_drop:
            X = X.dropna(subset=["total_charges"])
            churn = churn.loc[X.index]
            print(f"[preprocess] Dropped {n_drop} rows with NaN total_charges. "
                  f"Remaining rows: {len(X)}")

    n_nan = X.isnull().sum().sum()
    if n_nan:
        # remaining NaN cells are binary service columns — "no service" -> 0
        print(f"[preprocess] Filling {n_nan} NaN cells with 0 "
              "(service-not-applicable columns).")
        X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[preprocess] Scaled feature matrix shape: {X_scaled.shape}")

    return X_scaled, churn, list(X.columns)


# ── 4. Sweep ────────────────────────────────────────────────────────────────

def run_sweep(
    X: np.ndarray,
    k_max: int = 10,
    subsample: int = 2000,
    random_state: int = 42,
) -> tuple[list[int], list[float], list[float]]:
    """
    Fit KMeans for k = 2 … k_max, collecting inertia and silhouette score.

    Silhouette is evaluated on a random subsample of *subsample* rows when
    the dataset exceeds that size (avoids O(n²) silhouette cost on large data).

    Returns
    -------
    ks          : cluster counts tested
    inertias    : total within-cluster sum of squares per k
    silhouettes : average silhouette coefficient per k
    """
    ks, inertias, silhouettes = [], [], []

    n = len(X)
    use_sub = n > subsample
    if use_sub:
        print(f"\n[run_sweep] n={n} > {subsample}: silhouette computed on a "
              f"random subsample of {subsample} rows for speed.")
        rng = np.random.default_rng(random_state)
        sub_idx = rng.choice(n, size=subsample, replace=False)

    col_w = max(len("inertia"), 14)
    print(f"\n[run_sweep] Sweeping k = 2 ... {k_max}\n")
    header = f"{'k':>4}  {'inertia':>{col_w}}  {'silhouette':>12}"
    print(header)
    print("-" * len(header))

    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)

        X_sil    = X[sub_idx]      if use_sub else X
        lbl_sil  = labels[sub_idx] if use_sub else labels
        sil = silhouette_score(X_sil, lbl_sil)

        ks.append(k)
        inertias.append(km.inertia_)
        silhouettes.append(sil)

        print(f"{k:>4}  {km.inertia_:>{col_w}.2f}  {sil:>12.4f}")

    return ks, inertias, silhouettes


# ── 5. Detect elbow ─────────────────────────────────────────────────────────

def detect_elbow(ks: list[int], inertias: list[float]) -> int:
    """
    Find the elbow in the inertia-vs-k curve.

    Uses kneed.KneeLocator (convex, decreasing) when available; otherwise
    applies the max-perpendicular-distance-to-chord method:
      - normalise both axes to [0, 1]
      - compute each point's perpendicular distance to the line from
        (k_min, inertia_max) to (k_max, inertia_min)
      - return the k with the largest distance

    Returns
    -------
    elbow_k : int
    """
    if _KNEED:
        kl = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
        elbow_k = kl.knee or ks[0]
        print(f"\n[detect_elbow] kneed.KneeLocator -> elbow at k = {elbow_k}")
        return elbow_k

    # max-chord-distance fallback
    x = np.array(ks,       dtype=float)
    y = np.array(inertias, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min())
    y_n = (y - y.min()) / (y.max() - y.min())

    p1, p2   = np.array([x_n[0], y_n[0]]), np.array([x_n[-1], y_n[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    dists = []
    for xi, yi in zip(x_n, y_n):
        diff = p1 - np.array([xi, yi])
        # explicit 2D cross product (scalar): avoids np.cross deprecation in NumPy 2.0
        cross_val = line_vec[0] * diff[1] - line_vec[1] * diff[0]
        dists.append(abs(cross_val) / line_len)
    elbow_k = ks[int(np.argmax(dists))]
    print(f"\n[detect_elbow] max-chord-distance -> elbow at k = {elbow_k}")
    return elbow_k


# ── 6. Plot ─────────────────────────────────────────────────────────────────

def plot_results(
    ks: list[int],
    inertias: list[float],
    silhouettes: list[float],
    elbow_k: int,
    best_sil_k: int,
    out_dir: Path,
) -> None:
    """
    Produce and save the elbow plot and the silhouette plot.

    Files written
    -------------
    elbow_plot.png      — inertia vs k, elbow marked
    silhouette_plot.png — avg silhouette vs k, best k marked
    """
    elbow_inertia  = inertias[ks.index(elbow_k)]
    best_sil_score = silhouettes[ks.index(best_sil_k)]

    # ── elbow plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, marker="o", linewidth=2, color="steelblue", label="Inertia")
    ax.axvline(elbow_k, color="tomato", linestyle="--", linewidth=1.5,
               label=f"Elbow  k = {elbow_k}")
    ax.annotate(
        f"k = {elbow_k}",
        xy=(elbow_k, elbow_inertia),
        xytext=(elbow_k + 0.25, elbow_inertia * 1.04),
        fontsize=10, color="tomato",
        arrowprops=dict(arrowstyle="->", color="tomato", lw=1.2),
    )
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Total Within-Cluster Variation (Inertia)", fontsize=12)
    ax.set_title("Elbow Method — K-Means Clustering", fontsize=14)
    ax.set_xticks(ks)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout()
    elbow_path = out_dir / "elbow_plot.png"
    fig.savefig(elbow_path, dpi=150)
    plt.close(fig)
    print(f"\n[plot_results] Saved -> {elbow_path}")

    # ── silhouette plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, silhouettes, marker="o", linewidth=2,
            color="mediumseagreen", label="Avg silhouette score")
    ax.axvline(best_sil_k, color="darkorange", linestyle="--", linewidth=1.5,
               label=f"Best k = {best_sil_k}  (score = {best_sil_score:.4f})")
    ax.annotate(
        f"k = {best_sil_k}",
        xy=(best_sil_k, best_sil_score),
        xytext=(best_sil_k + 0.25, best_sil_score - 0.012),
        fontsize=10, color="darkorange",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2),
    )
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Average Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Analysis — K-Means Clustering", fontsize=14)
    ax.set_xticks(ks)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout()
    sil_path = out_dir / "silhouette_plot.png"
    fig.savefig(sil_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] Saved -> {sil_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-Means elbow + silhouette sweep on the churn dataset."
    )
    parser.add_argument("--k_max",     type=int,  default=10,
                        help="Max clusters to test (default: 10)")
    parser.add_argument("--subsample", type=int,  default=2000,
                        help="Max rows for silhouette scoring (default: 2000)")
    parser.add_argument("--data",      type=Path, default=DEFAULT_DATA,
                        help="Path to cleaned CSV")
    args = parser.parse_args()

    df = load_data(args.data)
    inspect_data(df)

    X_scaled, churn, feature_names = preprocess(df)

    ks, inertias, silhouettes = run_sweep(
        X_scaled, k_max=args.k_max, subsample=args.subsample
    )

    elbow_k    = detect_elbow(ks, inertias)
    best_sil_k = ks[int(np.argmax(silhouettes))]

    print(f"\n{'='*55}")
    print(f"  Auto-detected elbow k   : {elbow_k}")
    print(f"  Best silhouette k       : {best_sil_k}")
    if elbow_k != best_sil_k:
        print(
            f"\n  Note: the two methods disagree.\n"
            f"  Elbow (k={elbow_k}) minimises the marginal gain in within-cluster\n"
            f"  compactness; silhouette (k={best_sil_k}) maximises the ratio of\n"
            f"  inter- to intra-cluster separation. Inspect both plots and apply\n"
            f"  domain knowledge to choose."
        )
    print(f"{'='*55}\n")

    plot_results(ks, inertias, silhouettes, elbow_k, best_sil_k, PLOTS_DIR)


if __name__ == "__main__":
    main()
