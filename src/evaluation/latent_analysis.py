"""Latent space analysis: t-SNE, K-means, silhouette score."""

from __future__ import annotations

import os
import warnings

import torch
from torch import Tensor

try:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def analyze_latent_space(
    latents: Tensor,
    labels: Tensor,
    output_dir: str,
) -> dict:
    """Analyze a set of latent vectors using dimensionality reduction and clustering.

    Performs:
    - t-SNE projection to 2D
    - K-means clustering (k=2)
    - Silhouette score
    - Centroid L2 distance
    - Optional scatter plot saved to output_dir

    Parameters
    ----------
    latents : Tensor
        Shape ``(N, latent_dim)`` or ``(N, seq_len, latent_dim)`` — mean-pooled
        if 3D.
    labels : Tensor
        Integer labels of shape ``(N,)``.
    output_dir : str
        Directory to save the t-SNE plot if matplotlib is available.

    Returns
    -------
    dict
        Keys: silhouette_score, centroid_distance, n_samples, n_dims,
              plot_path (str or None).
    """
    if not _SKLEARN_AVAILABLE:
        warnings.warn(
            "scikit-learn not installed — skipping latent analysis. "
            "Install with: uv add scikit-learn",
            ImportWarning,
            stacklevel=2,
        )
        return {}

    # Mean-pool if 3D
    if latents.dim() == 3:
        latents = latents.mean(dim=1)

    latents_np = latents.detach().cpu().float().numpy()
    labels_np = labels.detach().cpu().numpy()

    n_samples, n_dims = latents_np.shape

    # t-SNE to 2D
    perplexity = min(30, max(5, n_samples // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded = tsne.fit_transform(latents_np)  # (N, 2)

    # K-means (k=2)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latents_np)

    # Silhouette score (needs at least 2 distinct clusters)
    unique_clusters = set(cluster_labels.tolist())
    if len(unique_clusters) >= 2:
        sil_score = float(silhouette_score(latents_np, cluster_labels))
    else:
        sil_score = 0.0

    # Centroid L2 distance
    c0 = kmeans.cluster_centers_[0]
    c1 = kmeans.cluster_centers_[1]
    centroid_dist = float(((c0 - c1) ** 2).sum() ** 0.5)

    # Plot
    plot_path: str | None = None
    if _MATPLOTLIB_AVAILABLE:
        try:
            os.makedirs(output_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                embedded[:, 0], embedded[:, 1],
                c=labels_np, cmap="coolwarm", alpha=0.6, s=20,
            )
            ax.set_title("t-SNE of Latent Space")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            plt.colorbar(scatter, ax=ax, label="Label")
            plot_path = os.path.join(output_dir, "latent_tsne.png")
            fig.savefig(plot_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Failed to save t-SNE plot: {exc}", stacklevel=2)
            plot_path = None

    return {
        "silhouette_score": sil_score,
        "centroid_distance": centroid_dist,
        "n_samples": n_samples,
        "n_dims": n_dims,
        "plot_path": plot_path,
    }
