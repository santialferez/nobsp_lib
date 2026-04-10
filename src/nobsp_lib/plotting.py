"""Plotting helpers for the simple NObSP wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: str | Path | None) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_training_history(
    losses: list[float],
    title: str = "Training loss",
    save_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, linewidth=2, color="#1f77b4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    _ensure_parent(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_tabular_feature_curves(
    X: np.ndarray,
    contributions: np.ndarray,
    feature_names: list[str] | None = None,
    output_index: int = 0,
    max_features: int = 6,
    save_path: str | Path | None = None,
):
    X = np.asarray(X)
    contributions = np.asarray(contributions)
    if contributions.ndim == 2:
        contributions = contributions[..., None]

    n_features = min(X.shape[1], max_features)
    feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]
    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for idx in range(n_rows * n_cols):
        ax = axes.flat[idx]
        if idx >= n_features:
            ax.axis("off")
            continue
        sort_idx = np.argsort(X[:, idx])
        contrib = contributions[:, idx, output_index]
        contrib = contrib - contrib.mean()
        ax.plot(X[sort_idx, idx], contrib[sort_idx], color="#d62728", linewidth=2)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(feature_names[idx])
        ax.set_xlabel("Feature value")
        ax.set_ylabel("Contribution")
        ax.grid(alpha=0.3, linestyle=":")

    fig.tight_layout()
    _ensure_parent(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_importance_heatmap(
    importance: np.ndarray,
    feature_names: list[str] | None = None,
    output_names: list[str] | None = None,
    title: str = "NObSP feature importance",
    save_path: str | Path | None = None,
):
    importance = np.asarray(importance)
    if importance.ndim == 1:
        importance = importance[:, None]

    feature_names = feature_names or [f"x{i}" for i in range(importance.shape[0])]
    output_names = output_names or [f"out_{i}" for i in range(importance.shape[1])]

    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * importance.shape[1], 2 + 0.5 * importance.shape[0]))
    im = ax.imshow(importance, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(output_names)))
    ax.set_xticklabels(output_names)
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _ensure_parent(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    cmap: str = "jet",
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    heatmap = np.asarray(heatmap, dtype=np.float32)

    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[0] in {1, 3}:
        image_rgb = np.moveaxis(image, 0, -1)
    else:
        image_rgb = image

    image_rgb = image_rgb - image_rgb.min()
    if image_rgb.max() > 0:
        image_rgb = image_rgb / image_rgb.max()

    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_rgb = plt.get_cmap(cmap)(heatmap)[..., :3]
    return np.clip((1 - alpha) * image_rgb + alpha * heatmap_rgb, 0.0, 1.0)


def plot_vision_gallery(
    images: np.ndarray,
    results: list[dict],
    class_names: list[str] | None = None,
    max_images: int = 6,
    heatmap_mode: Literal["positive", "negative", "mixed"] = "positive",
    overlay_alpha: float = 0.65,
    save_path: str | Path | None = None,
):
    image_count = min(len(results), max_images)
    fig, axes = plt.subplots(image_count, 2, figsize=(8, 3 * image_count))
    axes = np.atleast_2d(axes)
    title_map = {
        "positive": "NObSP positive",
        "negative": "NObSP negative",
        "mixed": "NObSP mixed",
    }

    for idx in range(image_count):
        image = np.asarray(images[idx])
        result = results[idx]
        heatmap = select_vision_heatmap(result, mode=heatmap_mode)
        overlay = overlay_heatmap(image, heatmap, alpha=overlay_alpha)
        pred_idx = int(result["predicted_class"])
        target_idx = int(result["target_class"])
        pred_name = class_names[pred_idx] if class_names else str(pred_idx)
        target_name = class_names[target_idx] if class_names else str(target_idx)

        axes[idx, 0].imshow(np.squeeze(image), cmap="gray" if np.squeeze(image).ndim == 2 else None)
        axes[idx, 0].set_title(f"Original\npred={pred_name}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(overlay)
        axes[idx, 1].set_title(f"{title_map[heatmap_mode]} heatmap\ntarget={target_name}")
        axes[idx, 1].axis("off")

    fig.tight_layout()
    _ensure_parent(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_vision_split_gallery(
    images: np.ndarray,
    results: list[dict],
    class_names: list[str] | None = None,
    max_images: int = 6,
    overlay_alpha: float = 0.65,
    save_path: str | Path | None = None,
):
    image_count = min(len(results), max_images)
    fig, axes = plt.subplots(image_count, 3, figsize=(12, 3 * image_count))
    axes = np.atleast_2d(axes)

    for idx in range(image_count):
        image = np.asarray(images[idx])
        result = results[idx]
        positive_overlay = overlay_heatmap(
            image,
            select_vision_heatmap(result, mode="positive"),
            alpha=overlay_alpha,
        )
        negative_overlay = overlay_heatmap(
            image,
            select_vision_heatmap(result, mode="negative"),
            alpha=overlay_alpha,
        )
        pred_idx = int(result["predicted_class"])
        target_idx = int(result["target_class"])
        pred_name = class_names[pred_idx] if class_names else str(pred_idx)
        target_name = class_names[target_idx] if class_names else str(target_idx)

        axes[idx, 0].imshow(np.squeeze(image), cmap="gray" if np.squeeze(image).ndim == 2 else None)
        axes[idx, 0].set_title(f"Original\npred={pred_name}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(positive_overlay)
        axes[idx, 1].set_title(f"NObSP positive\ntarget={target_name}")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(negative_overlay)
        axes[idx, 2].set_title(f"NObSP negative\ntarget={target_name}")
        axes[idx, 2].axis("off")

    fig.tight_layout()
    _ensure_parent(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def select_vision_heatmap(
    result: dict,
    mode: Literal["positive", "negative", "mixed"] = "positive",
) -> np.ndarray:
    if mode == "positive":
        heatmap = result.get("heatmap_positive", result.get("heatmap"))
    elif mode == "negative":
        heatmap = result.get("heatmap_negative", result.get("heatmap"))
    else:
        heatmap = result.get("heatmap_mixed", result.get("heatmap"))

    if heatmap is None:
        raise KeyError(f"Heatmap mode '{mode}' is not available in the result dictionary.")
    return np.asarray(heatmap)
