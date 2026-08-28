from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _to_numpy(value):
    """Convert a tensor or array-like object to a NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def TSNE_plot(
    features,
    y_true,
    y_pred=None,
    save_dir="./Classification_Figure/TSNE",
    random_state=42,
    perplexity=30.0,
):
    """Project sample-level latent features to 2-D and save t-SNE figures.

    The same embedding is used for the ground-truth and prediction panels, so
    their class distributions can be compared directly.
    """
    features = _to_numpy(features)
    y_true = _to_numpy(y_true).reshape(-1)
    y_pred = None if y_pred is None else _to_numpy(y_pred).reshape(-1)

    if features.ndim != 2:
        raise ValueError(
            f"features must have shape [N, D], but got {features.shape}"
        )

    sample_num = features.shape[0]
    if sample_num < 3:
        raise ValueError("t-SNE requires at least three samples")
    if y_true.shape[0] != sample_num:
        raise ValueError(
            "Feature/label size mismatch: "
            f"features={sample_num}, y_true={y_true.shape[0]}"
        )
    if y_pred is not None and y_pred.shape[0] != sample_num:
        raise ValueError(
            "Feature/prediction size mismatch: "
            f"features={sample_num}, y_pred={y_pred.shape[0]}"
        )
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinite values")

    # PCA reduces the runtime and noise of t-SNE for high-dimensional latent
    # representations while preserving one row per sample.
    pca_dim = min(50, features.shape[1], sample_num - 1)
    if features.shape[1] > pca_dim and pca_dim >= 2:
        features = PCA(
            n_components=pca_dim,
            random_state=random_state,
        ).fit_transform(features)

    # sklearn requires perplexity to be strictly smaller than sample_num.
    effective_perplexity = min(float(perplexity), (sample_num - 1) / 3.0)
    effective_perplexity = max(1.0, effective_perplexity)
    embedding = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate=200.0,
        random_state=random_state,
    ).fit_transform(features)

    labels = np.unique(
        y_true if y_pred is None else np.concatenate([y_true, y_pred])
    )
    cmap = plt.get_cmap("tab10" if labels.size <= 10 else "tab20")
    color_by_label = {
        label: cmap(index / max(1, labels.size - 1))
        for index, label in enumerate(labels)
    }

    panel_num = 1 if y_pred is None else 2
    fig, axes = plt.subplots(
        1,
        panel_num,
        figsize=(7 * panel_num, 6),
        squeeze=False,
    )

    def draw_panel(ax, panel_labels, title):
        for label in labels:
            mask = panel_labels == label
            if not np.any(mask):
                continue
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=12,
                alpha=0.75,
                color=color_by_label[label],
                label=f"Class {label}",
                linewidths=0,
            )
        ax.set_title(title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best", frameon=False)

    draw_panel(axes[0, 0], y_true, "Ground-truth labels")
    if y_pred is not None:
        draw_panel(axes[0, 1], y_pred, "Predicted labels")

    fig.tight_layout()
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "TSNE.png"
    pdf_path = output_dir / "TSNE.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"t-SNE figures saved to: {png_path} and {pdf_path}")
    return embedding


__all__ = ["TSNE_plot"]
