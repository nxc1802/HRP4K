from __future__ import annotations

from pathlib import Path


SPLITS = ("train", "valid", "test")


def image_path(data_dir: Path, split: str, file_name: str) -> Path:
    return data_dir / split / "images" / Path(file_name).name
