import io
import tarfile
from pathlib import Path

import pytest

from inat_project.paths import (
    extract_configured_dataset,
    find_project_root,
    get_project_paths,
    load_env_file,
)


def _project_root(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    return tmp_path


def _write_dataset_archive(path: Path) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for member_name in (
            "selected_images/train_mini/example_class/train.jpg",
            "selected_images/val/example_class/test.jpg",
        ):
            content = b"not a real image; archive layout is what this test covers"
            member = tarfile.TarInfo(member_name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def test_env_paths_are_repository_relative_and_shell_overrides_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = _project_root(tmp_path)
    env_file = root / ".env"
    env_file.write_text(
        "INAT_DATA_ARCHIVE=archives/selected_images.tar.gz\n"
        "INAT_DATA_DIR=external/data\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("INAT_DATA_ARCHIVE", raising=False)
    monkeypatch.delenv("INAT_DATA_DIR", raising=False)

    paths = get_project_paths(root)

    assert paths.project_root == root.resolve()
    assert paths.data_archive == (root / "archives/selected_images.tar.gz").resolve()
    assert paths.data_root == (root / "external/data").resolve()

    monkeypatch.setenv("INAT_DATA_DIR", "from-shell")
    paths = get_project_paths(root)
    assert paths.data_root == (root / "from-shell").resolve()


def test_extract_configured_dataset_accepts_one_wrapper_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = _project_root(tmp_path)
    archive = root / "selected_images.tar.gz"
    _write_dataset_archive(archive)
    (root / ".env").write_text(
        f"INAT_DATA_ARCHIVE={archive}\nINAT_DATA_DIR=data/inat500\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("INAT_DATA_ARCHIVE", raising=False)
    monkeypatch.delenv("INAT_DATA_DIR", raising=False)

    paths = get_project_paths(root)
    assert extract_configured_dataset(paths) is True
    assert paths.dataset_ready
    assert (paths.data_root / "train_mini/example_class/train.jpg").is_file()
    assert (paths.data_root / "val/example_class/test.jpg").is_file()
    assert extract_configured_dataset(paths) is False


def test_env_parser_rejects_invalid_entries(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("this is not an environment assignment\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        load_env_file(env_file)


def test_find_project_root_searches_upwards(tmp_path: Path):
    root = _project_root(tmp_path)
    nested = root / "notebooks" / "deep"
    nested.mkdir(parents=True)

    assert find_project_root(nested) == root.resolve()
