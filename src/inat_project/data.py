from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _class_seed(seed: int, class_name: str) -> int:
    value = hashlib.sha256(f"{seed}:{class_name}".encode("utf-8")).digest()
    return int.from_bytes(value[:8], byteorder="big", signed=False)


def _split_one_class(
    images: list[Path],
    validation_count: int,
    seed: int,
    class_name: str,
) -> tuple[list[Path], list[Path]]:
    """Split one class without placing byte-identical images on both sides."""
    groups: dict[str, list[Path]] = {}
    for path in images:
        key = _sha256(path)
        if key not in groups:
            groups[key] = []
        groups[key].append(path)

    singletons = [group[0] for group in groups.values() if len(group) == 1]
    duplicates = [group for group in groups.values() if len(group) > 1]

    #random number generator seeded by class name to ensure reproducibility across runs
    rng = random.Random(_class_seed(seed, class_name))
    rng.shuffle(singletons)
    rng.shuffle(duplicates)

    if len(singletons) < validation_count:
        raise ValueError(
            f"{class_name} has only {len(singletons)} unique singleton images; "
            f"cannot form a leak-free validation set of {validation_count}."
        )

    validation = sorted(singletons[:validation_count])
    validation_set = set(validation)
    training = sorted(path for path in images if path not in validation_set)
    return training, validation


def build_split_manifests(
    data_root: str | Path,
    metadata_root: str | Path,
    *,
    seed: int = 42,
    expected_classes: int = 500,
    train_per_class: int = 40,
    validation_per_class: int = 10,
    test_per_class: int = 10,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Create the canonical train, validation and test CSV files.

    `train_mini` supplies training and validation images. `val` is kept as the
    held-out test set. Existing manifests are reused unless `overwrite=True`.
    """
    data_root = Path(data_root).expanduser().resolve()
    metadata_root = Path(metadata_root).expanduser().resolve()
    split_root = metadata_root / "splits"
    paths = {
        "train": split_root / "train.csv",
        "validation": split_root / "validation.csv",
        "test": split_root / "test.csv",
        "class_to_idx": metadata_root / "class_to_idx.json",
        "summary": metadata_root / "split_summary.json",
    }

    if not overwrite and all(path.exists() for path in paths.values()):
        return paths

    train_source = data_root / "train_mini"
    test_source = data_root / "val"
    if not train_source.is_dir() or not test_source.is_dir():
        raise FileNotFoundError(
            f"Expected both {train_source} and {test_source} to exist."
        )

    train_classes = sorted(path.name for path in train_source.iterdir() if path.is_dir())
    test_classes = sorted(path.name for path in test_source.iterdir() if path.is_dir())

    if train_classes != test_classes:
        only_train = sorted(set(train_classes) - set(test_classes))
        only_test = sorted(set(test_classes) - set(train_classes))
        raise ValueError(
            "Class folders differ between train_mini and val. "
            f"Only in train_mini: {only_train[:5]}; only in val: {only_test[:5]}"
        )
    if len(train_classes) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} classes, found {len(train_classes)}."
        )

    class_to_idx = {name: index for index, name in enumerate(train_classes)}
    rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    expected_source_count = train_per_class + validation_per_class

    for class_name in train_classes:
        class_index = class_to_idx[class_name]
        source_id = class_name.split("_", maxsplit=1)[0]
        source_images = _image_files(train_source / class_name)
        test_images = _image_files(test_source / class_name)

        if len(source_images) != expected_source_count:
            raise ValueError(
                f"{class_name}: expected {expected_source_count} train_mini images, "
                f"found {len(source_images)}."
            )
        if len(test_images) != test_per_class:
            raise ValueError(
                f"{class_name}: expected {test_per_class} test images, "
                f"found {len(test_images)}."
            )

        training, validation = _split_one_class(
            source_images,
            validation_count=validation_per_class,
            seed=seed,
            class_name=class_name,
        )
        if len(training) != train_per_class:
            raise AssertionError(
                f"{class_name}: split produced {len(training)} training images."
            )

        for split_name, split_images in (
            ("train", training),
            ("validation", validation),
            ("test", test_images),
        ):
            for image_path in split_images:
                rows[split_name].append(
                    {
                        "path": image_path.relative_to(data_root).as_posix(),
                        "class_name": class_name,
                        "class_idx": class_index,
                        "source_id": source_id,
                        "split": split_name,
                    }
                )

    split_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation", "test"):
        pd.DataFrame(rows[split_name]).to_csv(paths[split_name], index=False)

    with paths["class_to_idx"].open("w", encoding="utf-8") as handle:
        json.dump(class_to_idx, handle, indent=2)

    summary = {
        "seed": seed,
        "classes": len(class_to_idx),
        "train_images": len(rows["train"]),
        "validation_images": len(rows["validation"]),
        "test_images": len(rows["test"]),
        "train_per_class": train_per_class,
        "validation_per_class": validation_per_class,
        "test_per_class": test_per_class,
    }
    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return paths


def validate_manifests(
    metadata_root: str | Path,
    *,
    expected_classes: int = 500,
    expected_counts: dict[str, int] | None = None,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Check split sizes, optional image paths, and return class counts."""
    metadata_root = Path(metadata_root)
    expected_counts = expected_counts or {
        "train": 40,
        "validation": 10,
        "test": 10,
    }

    frames = []
    for split_name in ("train", "validation", "test"):
        frame = pd.read_csv(metadata_root / "splits" / f"{split_name}.csv")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)

    counts = (
        combined.groupby(["class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    if len(counts) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {len(counts)}.")
    for split_name, expected in expected_counts.items():
        if split_name not in counts or not counts[split_name].eq(expected).all():
            raise ValueError(f"Unexpected per-class counts in {split_name}.")
    if combined["path"].duplicated().any():
        duplicate_paths = combined.loc[combined["path"].duplicated(), "path"].head(5)
        raise ValueError(f"The manifests reuse image paths across splits: {list(duplicate_paths)}")
    if data_root is not None:
        data_root = Path(data_root)
        missing = [
            relative_path
            for relative_path in combined["path"]
            if not (data_root / relative_path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} manifest image paths are missing below {data_root}; "
                f"first missing path: {missing[0]}"
            )
    return counts


class ManifestDataset(Dataset):
    """Image dataset backed by one of the checked-in split CSV files."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        transform: Callable | None = None,
        return_path: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.frame = pd.read_csv(self.manifest_path)
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self.data_root / row["path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        label = int(row["class_idx"])
        if self.return_path:
            return image, label, str(row["path"])
        return image, label


def build_transforms(image_size: int = 224) -> tuple[Callable, Callable]:
    """Return the training and deterministic evaluation transforms."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def build_classical_transform(image_size: int = 128) -> Callable:
    """Return deterministic uint8 pixels for handcrafted feature extraction."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.PILToTensor(),
        ]
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    data_root: str | Path,
    metadata_root: str | Path,
    *,
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 2,
    seed: int = 42,
    pin_memory: bool = False,
    train_transform: Callable | None = None,
    evaluation_transform: Callable | None = None,
    return_paths: bool = False,
) -> dict[str, DataLoader]:
    """Create consistent manifest-backed loaders for project experiments.

    The default transforms reproduce the ResNet experiments. Classical feature
    pipelines can supply deterministic transforms while retaining the exact
    same manifests, labels, batching and worker seeding.
    """
    data_root = Path(data_root)
    split_root = Path(metadata_root) / "splits"
    default_train_transform, default_evaluation_transform = build_transforms(
        image_size
    )
    train_transform = train_transform or default_train_transform
    evaluation_transform = evaluation_transform or default_evaluation_transform

    datasets = {
        "train": ManifestDataset(
            split_root / "train.csv",
            data_root,
            train_transform,
            return_path=return_paths,
        ),
        "validation": ManifestDataset(
            split_root / "validation.csv",
            data_root,
            evaluation_transform,
            return_path=return_paths,
        ),
        "test": ManifestDataset(
            split_root / "test.csv",
            data_root,
            evaluation_transform,
            return_path=True,
        ),
    }

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        "persistent_workers": num_workers > 0,
    }
    return {
        "train": DataLoader(
            datasets["train"], shuffle=True, generator=generator, **common
        ),
        "validation": DataLoader(datasets["validation"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
