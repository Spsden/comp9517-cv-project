from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import rgb_to_hsv
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hinge_loss,
)
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from .evaluation import summarise_predictions


FEATURE_NAMES = ("hog", "hsv", "hog_hsv")


def _rgb_array(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a CHW uint8/float tensor or array to an HWC float RGB array."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a three-dimensional image, found {image.shape}.")
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, found {image.shape}.")
    image = image.astype(np.float32, copy=False)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def extract_hog(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Extract the project HOG descriptor from one RGB image."""
    rgb = _rgb_array(image)
    grayscale = (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    )
    return hog(
        grayscale,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)


def extract_hsv(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Extract normalised hue, saturation and value histograms."""
    hsv = rgb_to_hsv(_rgb_array(image))
    hue, _ = np.histogram(hsv[..., 0], bins=32, range=(0.0, 1.0))
    saturation, _ = np.histogram(hsv[..., 1], bins=16, range=(0.0, 1.0))
    value, _ = np.histogram(hsv[..., 2], bins=16, range=(0.0, 1.0))
    features = np.concatenate([hue, saturation, value]).astype(np.float32)
    features /= features.sum() + 1e-8
    return features


def extract_loader_features(loader, *, description: str = "features") -> dict:
    """Extract HOG and HSV matrices from a manifest-backed DataLoader."""
    hog_rows: list[np.ndarray] = []
    hsv_rows: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    started = time.perf_counter()

    for batch in tqdm(loader, desc=description):
        if len(batch) != 3:
            raise ValueError(
                "Classical feature extraction requires return_paths=True in "
                "create_dataloaders()."
            )
        images, batch_labels, batch_paths = batch
        for image in images:
            hog_rows.append(extract_hog(image))
            hsv_rows.append(extract_hsv(image))
        labels.extend(int(value) for value in batch_labels)
        paths.extend(str(value) for value in batch_paths)

    return {
        "hog": np.stack(hog_rows),
        "hsv": np.stack(hsv_rows),
        "labels": np.asarray(labels, dtype=np.int64),
        "paths": np.asarray(paths, dtype=str),
        "extraction_seconds": time.perf_counter() - started,
    }


def feature_matrix(extracted: dict, feature_name: str) -> np.ndarray:
    """Select HOG, HSV or their concatenation from extracted features."""
    if feature_name == "hog":
        return extracted["hog"]
    if feature_name == "hsv":
        return extracted["hsv"]
    if feature_name == "hog_hsv":
        return np.concatenate([extracted["hog"], extracted["hsv"]], axis=1)
    raise ValueError(f"Unknown feature set: {feature_name!r}")


def save_feature_cache(path: str | Path, extracted: dict) -> None:
    """Cache a split's handcrafted features for fast notebook reruns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        hog=extracted["hog"],
        hsv=extracted["hsv"],
        labels=extracted["labels"],
        paths=extracted["paths"],
        extraction_seconds=np.asarray(extracted["extraction_seconds"]),
    )


def load_feature_cache(path: str | Path) -> dict:
    """Load a cache written by :func:`save_feature_cache`."""
    with np.load(Path(path), allow_pickle=False) as cache:
        return {
            "hog": cache["hog"],
            "hsv": cache["hsv"],
            "labels": cache["labels"],
            "paths": cache["paths"],
            "extraction_seconds": float(cache["extraction_seconds"]),
        }


def make_linear_svc(*, c_value: float, seed: int, max_iter: int) -> Pipeline:
    """Build the scaled linear SVM used for every handcrafted feature set."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearSVC(
                    C=c_value,
                    dual="auto",
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def tune_linear_svc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    c_values: Iterable[float],
    seed: int = 42,
    max_iter: int = 5000,
) -> tuple[Pipeline, pd.DataFrame]:
    """Select C using validation macro F1 and return all tuning measurements."""
    best_model: Pipeline | None = None
    best_validation_f1 = -np.inf
    rows = []

    for c_value in c_values:
        model = make_linear_svc(c_value=float(c_value), seed=seed, max_iter=max_iter)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        validation_scores = model.decision_function(x_validation)
        validation_predictions = model.predict(x_validation)
        validation_seconds = time.perf_counter() - started

        training_scores = model.decision_function(x_train)
        training_predictions = model.predict(x_train)
        classes = model.named_steps["classifier"].classes_
        validation_f1 = f1_score(
            y_validation,
            validation_predictions,
            average="macro",
            zero_division=0,
        )
        rows.append(
            {
                "C": float(c_value),
                "fit_seconds": fit_seconds,
                "validation_seconds": validation_seconds,
                "training_accuracy": accuracy_score(y_train, training_predictions),
                "validation_accuracy": accuracy_score(
                    y_validation, validation_predictions
                ),
                "training_macro_f1": f1_score(
                    y_train,
                    training_predictions,
                    average="macro",
                    zero_division=0,
                ),
                "validation_macro_f1": validation_f1,
                "training_hinge_loss": hinge_loss(
                    y_train, training_scores, labels=classes
                ),
                "validation_hinge_loss": hinge_loss(
                    y_validation, validation_scores, labels=classes
                ),
            }
        )
        if validation_f1 > best_validation_f1:
            best_model = model
            best_validation_f1 = validation_f1

    if best_model is None:
        raise ValueError("At least one C value is required.")
    return best_model, pd.DataFrame(rows)


def make_sgd_hinge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    c_value: float,
    epochs: int,
    seed: int,
    description: str | None = None,
) -> Pipeline:
    """Fit a scaled linear hinge-loss classifier incrementally."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    alpha = 1.0 / (float(c_value) * len(y_train))
    classifier = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        shuffle=True,
        average=True,
        n_jobs=-1,
        random_state=seed,
    )
    classes = np.unique(y_train)
    progress = tqdm(
        range(1, epochs + 1),
        desc=description or f"SGD hinge (C={float(c_value):g})",
        unit="epoch",
        leave=False,
    )
    for _ in progress:
        classifier.partial_fit(x_scaled, y_train, classes=classes)
        progress.set_postfix(alpha=f"{alpha:.2e}")
    return Pipeline([("scaler", scaler), ("classifier", classifier)])


def tune_sgd_hinge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    c_values: Iterable[float],
    epochs: int = 10,
    seed: int = 42,
) -> tuple[Pipeline, pd.DataFrame]:
    """Select an SGD hinge classifier using validation macro F1."""
    best_model: Pipeline | None = None
    best_validation_f1 = -np.inf
    rows = []

    for c_value in tqdm(list(c_values), desc="C search", unit="model"):
        started = time.perf_counter()
        model = make_sgd_hinge(
            x_train,
            y_train,
            c_value=float(c_value),
            epochs=epochs,
            seed=seed,
        )
        fit_seconds = time.perf_counter() - started

        training_scores = model.decision_function(x_train)
        training_predictions = model.predict(x_train)
        started = time.perf_counter()
        validation_scores = model.decision_function(x_validation)
        validation_predictions = model.predict(x_validation)
        validation_seconds = time.perf_counter() - started
        classes = model.named_steps["classifier"].classes_
        validation_f1 = f1_score(
            y_validation,
            validation_predictions,
            average="macro",
            zero_division=0,
        )
        rows.append(
            {
                "C": float(c_value),
                "fit_seconds": fit_seconds,
                "validation_seconds": validation_seconds,
                "training_accuracy": accuracy_score(y_train, training_predictions),
                "validation_accuracy": accuracy_score(
                    y_validation, validation_predictions
                ),
                "training_macro_f1": f1_score(
                    y_train,
                    training_predictions,
                    average="macro",
                    zero_division=0,
                ),
                "validation_macro_f1": validation_f1,
                "training_hinge_loss": hinge_loss(
                    y_train, training_scores, labels=classes
                ),
                "validation_hinge_loss": hinge_loss(
                    y_validation, validation_scores, labels=classes
                ),
            }
        )
        if validation_f1 > best_validation_f1:
            best_model = model
            best_validation_f1 = validation_f1

    if best_model is None:
        raise ValueError("At least one C value is required.")
    return best_model, pd.DataFrame(rows)


def collect_svc_predictions(
    model: Pipeline,
    x_test: np.ndarray,
    y_test: np.ndarray,
    paths: Iterable[str],
) -> tuple[pd.DataFrame, float]:
    """Collect top-1/top-5 predictions in the common project table format."""
    started = time.perf_counter()
    scores = model.decision_function(x_test)
    predicted = model.predict(x_test)
    inference_seconds = time.perf_counter() - started

    classes = model.named_steps["classifier"].classes_
    top_k = min(5, len(classes))
    top_positions = np.argsort(scores, axis=1)[:, -top_k:][:, ::-1]
    top_labels = classes[top_positions]
    top_scores = np.take_along_axis(scores, top_positions, axis=1)

    rows = []
    for index, path in enumerate(paths):
        true_index = int(y_test[index])
        predicted_index = int(predicted[index])
        top_indices = [int(value) for value in top_labels[index]]
        rows.append(
            {
                "path": str(path),
                "true_idx": true_index,
                "predicted_idx": predicted_index,
                "decision_score": float(top_scores[index, 0]),
                "correct": predicted_index == true_index,
                "top5_correct": true_index in top_indices,
                "top5_indices": json.dumps(top_indices),
            }
        )
    return pd.DataFrame(rows), inference_seconds


def evaluate_linear_classifier(
    model: Pipeline,
    x_test: np.ndarray,
    y_test: np.ndarray,
    paths: Iterable[str],
    *,
    num_classes: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate a selected SVM using the same summary metrics as deep models."""
    predictions, inference_seconds = collect_svc_predictions(
        model, x_test, y_test, paths
    )
    metrics = summarise_predictions(
        predictions,
        num_classes=num_classes,
        inference_seconds=inference_seconds,
    )
    return predictions, metrics


def evaluate_linear_svc(
    model: Pipeline,
    x_test: np.ndarray,
    y_test: np.ndarray,
    paths: Iterable[str],
    *,
    num_classes: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backward-compatible alias for the common linear-model evaluator."""
    return evaluate_linear_classifier(
        model,
        x_test,
        y_test,
        paths,
        num_classes=num_classes,
    )


def save_model(model: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
