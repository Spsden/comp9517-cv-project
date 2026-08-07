"""Portable local paths and dataset-archive setup for the project.

The repository never stores the image archive.  Each developer instead creates
a local ``.env`` from ``.env.example`` and sets ``INAT_DATA_ARCHIVE`` to their
own archive location.  All other locations may be left at their portable,
repository-relative defaults.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


ENV_FILENAME = ".env"
DATASET_FOLDERS = ("train_mini", "val")


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the repository root by searching upwards for ``pyproject.toml``."""
    candidate = Path(start or Path.cwd()).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (directory / "src").is_dir():
            return directory
    raise FileNotFoundError(
        "Could not locate the project root. Run from this repository or pass "
        "--project-root explicitly."
    )


def load_env_file(path: str | Path, *, override: bool = False) -> dict[str, str]:
    """Load simple ``KEY=VALUE`` entries without requiring python-dotenv.

    Existing environment variables take precedence by default, which allows a
    CI system or a one-off shell invocation to override a local ``.env``.
    """
    path = Path(path)
    if not path.is_file():
        return {}

    loaded: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = (part.strip() for part in line.split("=", maxsplit=1))
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"{path}:{line_number}: invalid environment-variable name")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _resolve_configured_path(value: str | None, default: Path, root: Path) -> Path:
    path = Path(value).expanduser() if value else default
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """All local project paths after reading the optional repository ``.env``."""

    project_root: Path
    data_root: Path
    metadata_root: Path
    checkpoint_root: Path
    results_root: Path
    cache_root: Path
    data_archive: Path | None

    @property
    def dataset_ready(self) -> bool:
        return all((self.data_root / name).is_dir() for name in DATASET_FOLDERS)


def get_project_paths(
    project_root: str | Path | None = None,
    *,
    env_path: str | Path | None = None,
) -> ProjectPaths:
    """Return project paths configured by the local environment.

    Defaults are safe after a clone: ``data/inat500`` for extracted images and
    standard repository folders for metadata, checkpoints, results, and cache.
    ``INAT_DATA_ARCHIVE`` is optional until data extraction is requested.
    """
    root = find_project_root(project_root)
    load_env_file(env_path or root / ENV_FILENAME)

    archive_value = os.environ.get("INAT_DATA_ARCHIVE", "").strip()
    archive = _resolve_configured_path(archive_value, root, root) if archive_value else None
    return ProjectPaths(
        project_root=root,
        data_root=_resolve_configured_path(
            os.environ.get("INAT_DATA_DIR"), root / "data" / "inat500", root
        ),
        metadata_root=_resolve_configured_path(
            os.environ.get("INAT_METADATA_DIR"), root / "metadata", root
        ),
        checkpoint_root=_resolve_configured_path(
            os.environ.get("INAT_CHECKPOINT_DIR"), root / "checkpoints", root
        ),
        results_root=_resolve_configured_path(
            os.environ.get("INAT_RESULTS_DIR"), root / "results", root
        ),
        cache_root=_resolve_configured_path(
            os.environ.get("INAT_CACHE_DIR"), root / "cache", root
        ),
        data_archive=archive,
    )


def _safe_member_path(destination: Path, member_name: str) -> Path:
    member_path = PurePosixPath(member_name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"Unsafe archive member: {member_name!r}")
    target = destination.joinpath(*member_path.parts).resolve()
    if target != destination and destination not in target.parents:
        raise ValueError(f"Archive member escapes extraction directory: {member_name!r}")
    return target


def _extract_tar_safely(archive_path: Path, destination: Path) -> None:
    """Extract regular files/directories only, rejecting traversal and links."""
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            target = _safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(
                    f"Unsupported archive member {member.name!r}; "
                    "links and special files are not accepted."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read archive member {member.name!r}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _find_dataset_root(extraction_root: Path) -> Path:
    """Find a directory containing the expected top-level dataset folders."""
    candidates = [extraction_root]
    candidates.extend(path for path in extraction_root.iterdir() if path.is_dir())
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in DATASET_FOLDERS):
            return candidate
    expected = " and ".join(DATASET_FOLDERS)
    raise ValueError(
        f"The archive does not contain both {expected} directories at its root "
        "or one level below it."
    )


def extract_configured_dataset(paths: ProjectPaths) -> bool:
    """Extract the configured archive if necessary and return whether work occurred.

    Extraction is staged next to the configured data directory.  The final data
    directory is only populated after the archive shape has been validated.
    Existing partial directories are never overwritten.
    """
    if paths.dataset_ready:
        return False
    if paths.data_archive is None:
        raise FileNotFoundError(
            "Dataset is not extracted and INAT_DATA_ARCHIVE is unset. Copy "
            ".env.example to .env and set the archive path."
        )
    if not paths.data_archive.is_file():
        raise FileNotFoundError(
            f"Configured dataset archive does not exist: {paths.data_archive}"
        )
    if paths.data_root.exists() and any(paths.data_root.iterdir()):
        raise FileExistsError(
            f"Refusing to mix an archive into non-empty data directory "
            f"{paths.data_root}. Remove or relocate that partial directory first."
        )

    paths.data_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{paths.data_root.name}-extract-", dir=paths.data_root.parent
    ) as temporary_directory:
        staging_root = Path(temporary_directory)
        _extract_tar_safely(paths.data_archive, staging_root)
        source_root = _find_dataset_root(staging_root)
        paths.data_root.mkdir(parents=True, exist_ok=True)
        for folder_name in DATASET_FOLDERS:
            shutil.move(str(source_root / folder_name), paths.data_root / folder_name)
    return True


def require_dataset(paths: ProjectPaths) -> Path:
    """Return a usable data root or explain the local setup command."""
    if not paths.dataset_ready:
        raise FileNotFoundError(
            f"Dataset folders are missing below {paths.data_root}. Run "
            "`inat-prepare` after configuring .env."
        )
    return paths.data_root
