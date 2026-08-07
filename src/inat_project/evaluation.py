from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm.auto import tqdm


@torch.inference_mode()
def collect_predictions(model, loader, device: torch.device) -> tuple[pd.DataFrame, float]:
    model.eval()
    records = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()

    for batch in tqdm(loader, desc="test", leave=False):
        if len(batch) == 3:
            images, labels, paths = batch
        else:
            images, labels = batch
            paths = [""] * len(labels)

        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = logits.softmax(dim=1)
        top_probabilities, top_indices = probabilities.topk(
            min(5, probabilities.shape[1]), dim=1
        )

        labels_np = labels.numpy()
        top_indices_np = top_indices.cpu().numpy()
        top_probabilities_np = top_probabilities.cpu().numpy()
        for index, path in enumerate(paths):
            true_index = int(labels_np[index])
            predicted_index = int(top_indices_np[index, 0])
            top5 = [int(value) for value in top_indices_np[index]]
            records.append(
                {
                    "path": path,
                    "true_idx": true_index,
                    "predicted_idx": predicted_index,
                    "confidence": float(top_probabilities_np[index, 0]),
                    "correct": predicted_index == true_index,
                    "top5_correct": true_index in top5,
                    "top5_indices": json.dumps(top5),
                }
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    return pd.DataFrame(records), elapsed


def summarise_predictions(
    predictions: pd.DataFrame,
    *,
    num_classes: int,
    inference_seconds: float,
) -> dict[str, float]:
    labels = np.arange(num_classes)
    y_true = predictions["true_idx"].to_numpy()
    y_pred = predictions["predicted_idx"].to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "top5_accuracy": float(predictions["top5_correct"].mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "test_images": int(len(predictions)),
        "inference_seconds": float(inference_seconds),
        "images_per_second": float(len(predictions) / inference_seconds),
    }


def confusion_data(
    predictions: pd.DataFrame, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    matrix = confusion_matrix(
        predictions["true_idx"],
        predictions["predicted_idx"],
        labels=np.arange(num_classes),
    )
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
    return matrix, normalised


def most_confused_pairs(
    matrix: np.ndarray,
    idx_to_class: dict[int, str],
    limit: int = 20,
) -> pd.DataFrame:
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    flat_indices = np.argsort(off_diagonal.ravel())[::-1]
    rows = []
    for flat_index in flat_indices:
        true_index, predicted_index = np.unravel_index(flat_index, matrix.shape)
        count = int(off_diagonal[true_index, predicted_index])
        if count == 0 or len(rows) >= limit:
            break
        rows.append(
            {
                "true_idx": int(true_index),
                "predicted_idx": int(predicted_index),
                "true_class": idx_to_class[int(true_index)],
                "predicted_class": idx_to_class[int(predicted_index)],
                "count": count,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "true_idx",
            "predicted_idx",
            "true_class",
            "predicted_class",
            "count",
        ],
    )


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
