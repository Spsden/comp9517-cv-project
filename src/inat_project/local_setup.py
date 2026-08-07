"""Command-line setup for a portable local checkout."""

from __future__ import annotations

import argparse

from .data import build_split_manifests, validate_manifests
from .paths import extract_configured_dataset, get_project_paths, require_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and validate the local iNaturalist project dataset."
    )
    parser.add_argument(
        "--project-root", help="Repository root (default: search from current directory)."
    )
    parser.add_argument("--env-file", help="Alternative .env file.")
    parser.add_argument(
        "--rebuild-manifests",
        action="store_true",
        help="Regenerate metadata/splits from the extracted dataset using seed 42.",
    )
    parser.add_argument(
        "--show-paths", action="store_true", help="Print resolved locations and exit."
    )
    args = parser.parse_args()

    paths = get_project_paths(args.project_root, env_path=args.env_file)
    print(f"Project:     {paths.project_root}")
    print(f"Data:        {paths.data_root}")
    print(f"Archive:     {paths.data_archive or '(not configured)'}")
    print(f"Checkpoints: {paths.checkpoint_root}")
    print(f"Results:     {paths.results_root}")
    if args.show_paths:
        return

    extracted = extract_configured_dataset(paths)
    print("Extracted dataset." if extracted else "Dataset already extracted.")
    data_root = require_dataset(paths)
    build_split_manifests(
        data_root,
        paths.metadata_root,
        overwrite=args.rebuild_manifests,
    )
    counts = validate_manifests(paths.metadata_root, data_root=data_root)
    print(f"Validated {len(counts)} classes and {int(counts.to_numpy().sum())} images.")
    print("Open a notebook from notebooks/ and run its first setup cell.")


if __name__ == "__main__":
    main()
