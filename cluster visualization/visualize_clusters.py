"""
visualize_clusters.py
K-Means cluster visualization and profiling on the cleaned Telco customer
churn dataset.

Produces:
  clusters_pca_2d.png        -- static matplotlib scatter (PCA 2D)
  clusters_pca_3d.html       -- interactive plotly scatter (PCA 3D)
  cluster_profile_heatmap.png -- z-scored feature heatmap per cluster
  cluster_profiles.csv       -- full per-cluster mean profile table

Dependencies: pandas, scikit-learn, matplotlib, plotly
Run:          python visualize_clusters.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless-safe; swap to TkAgg for an interactive window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent   # project root
DATA_PATH    = ROOT / "data" / "processed" / "cleaned_telco_customer_churn.csv"
OUT_DIR      = ROOT / "cluster plots"
OUT_DIR.mkdir(exist_ok=True)
N_CLUSTERS   = 4
RANDOM_STATE = 42
# ---------------------------------------------------------------------------


def load_and_clean(path: Path) -> pd.DataFrame:
    """
    Load CSV and handle missing values per data notes:
      - total_charges : drop the 11 NaN rows (tenure=0 new customers).
      - all others    : fill with 0 (service-not-applicable binary columns).

    'churn' is kept in the returned DataFrame so it can be used as a
    post-hoc label. It is separated from the clustering features in main()
    so that KMeans clusters on pure customer behaviour; churn is then
    re-attached only during profiling to see how it distributes across
    the behavioural segments.
    """
    df = pd.read_csv(path)

    df = df.dropna(subset=["total_charges"])   # 11 rows removed
    df = df.fillna(0)                          # remaining NaNs -> 0

    print(f"[load] {df.shape[0]} rows x {df.shape[1]} features after cleaning")
    return df


def fit_kmeans(X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit KMeans(k=4) and return (labels, cluster_centers_in_scaled_space)."""
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    return labels, km.cluster_centers_


def print_stats(labels: np.ndarray, X_scaled: np.ndarray) -> None:
    """Print total rows, per-cluster sizes, and overall silhouette score."""
    n = len(labels)
    print(f"\nTotal rows clustered : {n}")
    print("Cluster sizes:")
    for c, cnt in zip(*np.unique(labels, return_counts=True)):
        print(f"  Cluster {c} : {cnt:>4} rows  ({cnt / n * 100:.1f}%)")
    sil = silhouette_score(X_scaled, labels, sample_size=2000,
                           random_state=RANDOM_STATE)
    print(f"Silhouette score (subsample=2000) : {sil:.4f}\n")


def plot_2d(X_scaled: np.ndarray, labels: np.ndarray,
            centroids: np.ndarray) -> None:
    """
    PCA 2D scatter colored by cluster label.
    Centroids projected through the same PCA and drawn as black X markers.
    Saved to clusters_pca_2d.png.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords   = pca.fit_transform(X_scaled)
    cent_2d  = pca.transform(centroids)
    var      = pca.explained_variance_ratio_ * 100

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 6))

    for cid in range(N_CLUSTERS):
        mask = labels == cid
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=8, alpha=0.5, color=cmap(cid), label=f"Cluster {cid}")

    # project centroids and overlay as large black X
    ax.scatter(cent_2d[:, 0], cent_2d[:, 1],
               s=220, marker="X", color="black", zorder=5, label="Centroid")

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)", fontsize=12)
    ax.set_title("K-Means Clusters (k=4) -- PCA 2D Projection", fontsize=14)
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_path = OUT_DIR / "clusters_pca_2d.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[2D] Saved -> {out_path}")


def plot_3d(X_scaled: np.ndarray, labels: np.ndarray,
            centroids: np.ndarray) -> None:
    """
    Interactive PCA 3D scatter built with plotly.
    Cluster labels are strings so plotly uses discrete colors.
    Centroids overlaid as black diamond markers.
    Saved to clusters_pca_3d.html and opened in the default browser.
    """
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    coords  = pca.fit_transform(X_scaled)
    cent_3d = pca.transform(centroids)
    var     = pca.explained_variance_ratio_ * 100

    axis_labels = {
        "PC1": f"PC1 ({var[0]:.1f}% var)",
        "PC2": f"PC2 ({var[1]:.1f}% var)",
        "PC3": f"PC3 ({var[2]:.1f}% var)",
    }

    df_plot = pd.DataFrame({
        "PC1":     coords[:, 0],
        "PC2":     coords[:, 1],
        "PC3":     coords[:, 2],
        # string label -> discrete colorscale
        "Cluster": [f"Cluster {l}" for l in labels],
    })

    fig = px.scatter_3d(
        df_plot,
        x="PC1", y="PC2", z="PC3",
        color="Cluster",
        opacity=0.7,
        title="K-Means Clusters (k=4) -- PCA 3D Projection",
        labels=axis_labels,
        hover_data={"Cluster": True},
    )
    fig.update_traces(marker=dict(size=2))   # small points so dense regions stay readable

    # centroids as larger black diamonds in a separate trace
    fig.add_trace(go.Scatter3d(
        x=cent_3d[:, 0],
        y=cent_3d[:, 1],
        z=cent_3d[:, 2],
        mode="markers",
        marker=dict(size=8, color="black", symbol="diamond"),
        name="Centroid",
        hovertext=[f"Centroid {i}" for i in range(N_CLUSTERS)],
        hoverinfo="text",
    ))

    out_path = OUT_DIR / "clusters_pca_3d.html"
    fig.write_html(str(out_path))
    print(f"[3D] Saved -> {out_path}")
    fig.show()


# Columns shown in the printed profile summary (subset of all 28 features).
PROFILE_COLS = [
    "churn", "tenure", "monthly_charges", "total_charges",
    "contract_month-to-month", "contract_one_year", "contract_two_year",
    "internet_service_dsl", "internet_service_fiber_optic", "internet_service_no",
    "payment_method_bank_transfer_(automatic)", "payment_method_credit_card_(automatic)",
    "payment_method_electronic_check", "payment_method_mailed_check",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "multiple_lines",
]


def profile_clusters(df_unscaled: pd.DataFrame, labels: np.ndarray,
                     churn: pd.Series) -> None:
    """
    Profile each cluster on unscaled behavioural features.
    churn is passed in separately (it was excluded from clustering) and
    re-attached here purely as an outcome variable to inspect distribution.

    Steps
    -----
    1. Attach cluster labels + churn and compute per-cluster means.
    2. Print a summary table (size, churn rate, key feature means/proportions).
    3. Z-score the per-cluster means across clusters to surface unusual features.
    4. Plot a heatmap of z-scores annotated with raw means.
    5. Print the top-5 most distinguishing features per cluster.
    6. Save the full profile table to cluster_profiles.csv.
    """
    df_prof = df_unscaled.copy()
    df_prof["churn"]   = churn.values   # re-attach as outcome, not a cluster driver
    df_prof["cluster"] = labels

    feature_cols = list(df_unscaled.columns) + ["churn"]
    profile_cols = [c for c in PROFILE_COLS if c in df_prof.columns]

    # overall reference churn rate
    overall_churn = churn.mean()
    print(f"\nOverall churn rate (reference) : {overall_churn:.1%}\n")

    # per-cluster means across every feature
    cluster_means = df_prof.groupby("cluster")[feature_cols].mean()
    cluster_sizes = df_prof.groupby("cluster").size().rename("size")
    profile_table = pd.concat([cluster_sizes, cluster_means], axis=1)

    # --- printed summary ---
    sep = "=" * 62
    print(sep)
    print("Cluster profiles  (unscaled means / proportions)")
    print(sep)
    for cid in sorted(df_prof["cluster"].unique()):
        row = profile_table.loc[cid]
        n   = int(row["size"])
        pct = n / len(df_prof) * 100
        print(f"\n--- Cluster {cid}  (n={n}, {pct:.1f}% of customers) ---")
        for col in profile_cols:
            if col in row.index:
                print(f"  {col:<46} {row[col]:.3f}")

    # --- z-score profile: z = (cluster_mean - mean_across_clusters) / std_across_clusters ---
    z_profile = (
        cluster_means
        .subtract(cluster_means.mean(axis=0))
        .divide(cluster_means.std(axis=0) + 1e-9)   # eps avoids /0 on constant cols
    )

    # --- top-5 distinguishing features per cluster ---
    print(f"\n{sep}")
    print("Top-5 distinguishing features per cluster (largest |z-score|)")
    print(sep)
    for cid in sorted(z_profile.index):
        z_row = z_profile.loc[cid]
        top5  = z_row.abs().nlargest(5)
        print(f"\nCluster {cid}:")
        for feat in top5.index:
            z_val     = z_row[feat]
            direction = "above" if z_val > 0 else "below"
            print(f"  {feat:<46} z={z_val:+.2f}  ({direction} avg)")

    # --- heatmap on the PROFILE_COLS subset for readability ---
    _plot_profile_heatmap(z_profile[profile_cols], cluster_means[profile_cols])

    # --- save full profile to CSV ---
    csv_path = OUT_DIR / "cluster_profiles.csv"
    profile_table.to_csv(csv_path)
    print(f"\n[profile] Saved profile table -> {csv_path}")


def _plot_profile_heatmap(z_profile: pd.DataFrame,
                           raw_means: pd.DataFrame) -> None:
    """
    Heatmap: features as rows, clusters as columns.
    Cells are colored by z-score (RdBu_r, centered at 0) and annotated
    with the raw unscaled mean so the absolute values remain visible.
    Saved to cluster_profile_heatmap.png.
    """
    # transpose: rows = features, columns = clusters
    z_data   = z_profile.T
    raw_data = raw_means.T

    n_feat, n_clust = z_data.shape
    fig_h = max(8, n_feat * 0.42)

    fig, ax = plt.subplots(figsize=(n_clust * 2.4 + 1.5, fig_h))
    im = ax.imshow(z_data.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)

    # annotate each cell with the raw mean
    for ri in range(n_feat):
        for ci in range(n_clust):
            raw_val = raw_data.iloc[ri, ci]
            ax.text(ci, ri, f"{raw_val:.2f}",
                    ha="center", va="center", fontsize=7.5, color="black")

    ax.set_xticks(range(n_clust))
    ax.set_xticklabels([f"Cluster {c}" for c in z_profile.index], fontsize=11)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(z_data.index, fontsize=8.5)
    ax.set_title(
        "Cluster Profile Heatmap  (z-scored across clusters, raw means annotated)",
        fontsize=12, pad=12,
    )
    plt.colorbar(im, ax=ax, label="Z-score", shrink=0.55, pad=0.02)
    fig.tight_layout()

    out_path = OUT_DIR / "cluster_profile_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[profile] Saved heatmap -> {out_path}")


def main() -> None:
    df = load_and_clean(DATA_PATH)

    # separate churn before clustering — treat it as an outcome, not a feature
    churn        = df.pop("churn")
    df_unscaled  = df.copy()   # unscaled behavioural features, kept for profiling

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df.values.astype(float))

    labels, centroids = fit_kmeans(X_scaled)
    print_stats(labels, X_scaled)

    plot_2d(X_scaled, labels, centroids)
    plot_3d(X_scaled, labels, centroids)

    profile_clusters(df_unscaled, labels, churn)


if __name__ == "__main__":
    main()
