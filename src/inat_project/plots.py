from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_training_history(history: list[dict], save_path: str | Path | None = None):
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(frame["epoch"], frame["train_loss"], label="training")
    axes[0].plot(frame["epoch"], frame["validation_loss"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[0].legend()

    axes[1].plot(frame["epoch"], frame["train_top1"], label="training")
    axes[1].plot(frame["epoch"], frame["validation_top1"], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Top-1 accuracy", title="Accuracy")
    axes[1].legend()
    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    return figure


def plot_confusion_matrix(
    matrix: np.ndarray,
    save_path: str | Path | None = None,
):
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axis.set(
        xlabel="Predicted class index",
        ylabel="True class index",
        title="Normalised confusion matrix (500 species)",
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=220, bbox_inches="tight")
    return figure


def plot_confused_pairs(
    pairs: pd.DataFrame,
    save_path: str | Path | None = None,
    limit: int = 15,
):
    shown = pairs.head(limit).copy()
    shown["pair"] = shown.apply(
        lambda row: f"{row.true_idx} → {row.predicted_idx}", axis=1
    )
    figure, axis = plt.subplots(figsize=(8, max(4, 0.35 * len(shown))))
    sns.barplot(data=shown, x="count", y="pair", color="#4c78a8", ax=axis)
    axis.set(
        xlabel="Number of test images",
        ylabel="True → predicted class index",
        title="Most frequent off-diagonal confusions",
    )
    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    return figure

