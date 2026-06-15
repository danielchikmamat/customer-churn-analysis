"""
visualize_clusters.py
K-Means cluster visualization on the cleaned Telco customer churn dataset.

Produces:
  clusters_pca_2d.png  -- static matplotlib scatter (PCA 2D)
  clusters_pca_3d.html -- interactive plotly scatter (PCA 3D)

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
    Also drops the 'churn' target — clustering is unsupervised.
    """
    df = pd.read_csv(path)

    if "churn" in df.columns:
        df = df.drop(columns=["churn"])

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


def main() -> None:
    df = load_and_clean(DATA_PATH)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df.values.astype(float))

    labels, centroids = fit_kmeans(X_scaled)
    print_stats(labels, X_scaled)

    plot_2d(X_scaled, labels, centroids)
    plot_3d(X_scaled, labels, centroids)


if __name__ == "__main__":
    main()
