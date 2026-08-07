"""Run the HOG, HSV and HOG+HSV classical baselines.

The companion ``05_hog_hsv_classical.ipynb`` presents the same workflow with
explanatory cells. This script is useful for a non-interactive complete run.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from inat_project.classical import (  # noqa: E402
    evaluate_linear_classifier,
    extract_loader_features,
    feature_matrix,
    load_feature_cache,
    save_feature_cache,
    save_model,
    tune_sgd_hinge,
)
from inat_project.config import load_config  # noqa: E402
from inat_project.data import (  # noqa: E402
    build_classical_transform,
    create_dataloaders,
    validate_manifests,
)
from inat_project.evaluation import (  # noqa: E402
    confusion_data,
    most_confused_pairs,
    save_metrics,
)
from inat_project.plots import (  # noqa: E402
    plot_confused_pairs,
    plot_confusion_matrix,
)


DISPLAY_NAMES = {"hog": "HOG", "hsv": "HSV", "hog_hsv": "HOG + HSV"}


def load_features(loaders, cache_root: Path, use_cache: bool) -> dict[str, dict]:
    extracted = {}
    for split_name, loader in loaders.items():
        cache_path = cache_root / f"{split_name}_features.npz"
        if use_cache and cache_path.is_file():
            print(f"Loading cached {split_name} features from {cache_path}")
            extracted[split_name] = load_feature_cache(cache_path)
        else:
            extracted[split_name] = extract_loader_features(
                loader, description=f"{split_name} features"
            )
            if use_cache:
                save_feature_cache(cache_path, extracted[split_name])
    return extracted


def plot_tuning(tuning: pd.DataFrame, title: str, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(tuning["C"], tuning["training_hinge_loss"], marker="o", label="training")
    axes[0].plot(tuning["C"], tuning["validation_hinge_loss"], marker="o", label="validation")
    axes[0].set(xscale="log", xlabel="C", ylabel="Multiclass hinge loss", title="Loss")
    axes[0].legend()

    axes[1].plot(tuning["C"], 100 * tuning["training_macro_f1"], marker="o", label="training")
    axes[1].plot(tuning["C"], 100 * tuning["validation_macro_f1"], marker="o", label="validation")
    axes[1].set(xscale="log", xlabel="C", ylabel="Macro F1 (%)", title="Performance")
    axes[1].legend()

    axes[2].plot(tuning["C"], tuning["fit_seconds"], marker="o", label="training")
    axes[2].plot(tuning["C"], tuning["validation_seconds"], marker="o", label="validation inference")
    axes[2].set(xscale="log", yscale="log", xlabel="C", ylabel="Seconds", title="Runtime")
    axes[2].legend()
    figure.suptitle(f"{title} hyperparameter selection")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_feature_experiment(
    feature_name: str,
    extracted: dict[str, dict],
    config: dict,
    idx_to_class: dict[int, str],
    result_root: Path,
    checkpoint_root: Path,
) -> dict:
    display_name = DISPLAY_NAMES[feature_name]
    print(f"\nExperiment: {display_name}")
    x_train = feature_matrix(extracted["train"], feature_name)
    x_validation = feature_matrix(extracted["validation"], feature_name)
    x_test = feature_matrix(extracted["test"], feature_name)
    y_train = extracted["train"]["labels"]
    y_validation = extracted["validation"]["labels"]
    y_test = extracted["test"]["labels"]

    model, tuning = tune_sgd_hinge(
        x_train,
        y_train,
        x_validation,
        y_validation,
        c_values=config["model"]["c_values"],
        seed=config["seed"],
        epochs=config["model"]["epochs"],
    )
    selected_index = tuning["validation_macro_f1"].idxmax()
    selected = tuning.loc[selected_index]
    predictions, metrics = evaluate_linear_classifier(
        model,
        x_test,
        y_test,
        extracted["test"]["paths"],
        num_classes=config["data"]["num_classes"],
    )
    metrics.update(
        {
            "selected_C": float(selected["C"]),
            "validation_macro_f1": float(selected["validation_macro_f1"]),
            "feature_dimension": int(x_train.shape[1]),
            "selected_fit_seconds": float(selected["fit_seconds"]),
            "tuning_fit_seconds": float(tuning["fit_seconds"].sum()),
            "feature_extraction_seconds": float(
                sum(value["extraction_seconds"] for value in extracted.values())
            ),
        }
    )

    experiment_root = result_root / feature_name
    figure_root = experiment_root / "figures"
    experiment_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    predictions["true_class"] = predictions["true_idx"].map(idx_to_class)
    predictions["predicted_class"] = predictions["predicted_idx"].map(idx_to_class)
    predictions.to_csv(experiment_root / "test_predictions.csv", index=False)
    tuning.to_csv(experiment_root / "tuning_history.csv", index=False)
    save_metrics(metrics, experiment_root / "metrics.json")
    save_model(model, checkpoint_root / f"{feature_name}_sgd_hinge.joblib")

    matrix, normalised = confusion_data(
        predictions, config["data"]["num_classes"]
    )
    pairs = most_confused_pairs(matrix, idx_to_class, limit=30)
    pairs.to_csv(experiment_root / "confused_pairs.csv", index=False)
    plot_tuning(tuning, display_name, figure_root / "tuning.png")
    plt.close(
        plot_confusion_matrix(normalised, figure_root / "confusion_matrix.png")
    )
    plt.close(plot_confused_pairs(pairs, figure_root / "confused_pairs.png"))
    print(pd.Series(metrics).to_string())
    return metrics


def plot_model_comparison(summary: pd.DataFrame, result_root: Path) -> None:
    comparison_root = result_root / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    metric_frame = summary.melt(
        id_vars="model",
        value_vars=["top1_accuracy", "top5_accuracy", "macro_f1"],
        var_name="metric",
        value_name="value",
    )
    metric_frame["value"] *= 100
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(data=metric_frame, x="metric", y="value", hue="model", ax=axis)
    axis.set(xlabel="", ylabel="Test performance (%)", title="Classical feature comparison", ylim=(0, 100))
    figure.tight_layout()
    figure.savefig(comparison_root / "test_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=summary,
        x="selected_fit_seconds",
        y="top1_accuracy",
        hue="model",
        s=120,
        ax=axis,
    )
    axis.set(xlabel="Selected model fit time (seconds)", ylabel="Top-1 accuracy", title="Performance versus training time")
    figure.tight_layout()
    figure.savefig(comparison_root / "accuracy_vs_time.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "metadata")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results/classical")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints/classical")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    config = load_config(REPO_ROOT / "configs/hog_hsv.yaml")
    validate_manifests(
        args.metadata_root, expected_classes=config["data"]["num_classes"]
    )
    transform = build_classical_transform(config["data"]["image_size"])
    loaders = create_dataloaders(
        args.data_root,
        args.metadata_root,
        batch_size=config["data"]["batch_size"],
        image_size=config["data"]["image_size"],
        num_workers=config["data"]["num_workers"],
        seed=config["seed"],
        train_transform=transform,
        evaluation_transform=transform,
        return_paths=True,
    )
    cache_root = args.checkpoint_root / "feature_cache"
    extracted = load_features(
        loaders,
        cache_root,
        use_cache=bool(config["features"]["cache"] and not args.no_cache),
    )
    with (args.metadata_root / "class_to_idx.json").open(encoding="utf-8") as handle:
        class_to_idx = json.load(handle)
    idx_to_class = {index: name for name, index in class_to_idx.items()}

    rows = []
    for feature_name in config["features"]["names"]:
        metrics = run_feature_experiment(
            feature_name,
            extracted,
            config,
            idx_to_class,
            args.results_root,
            args.checkpoint_root,
        )
        rows.append({"model": DISPLAY_NAMES[feature_name], **metrics})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.results_root / "comparison.csv", index=False)
    plot_model_comparison(summary, args.results_root)

    runtime = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    with (args.results_root / "runtime.json").open("w", encoding="utf-8") as handle:
        json.dump(runtime, handle, indent=2)
    print("\nFinal comparison")
    print(
        summary[
            ["model", "top1_accuracy", "top5_accuracy", "macro_f1", "selected_C"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
