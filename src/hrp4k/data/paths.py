from __future__ import annotations

import os
import zipfile
from pathlib import Path

SPLITS = ("train", "valid", "test")

KAGGLE_CANDIDATE_PATHS = [
    Path("/kaggle/input/datasets/cuongnguyen1802/hrp4k-dataset-2/HRP4K/HRP4K"),
    Path("/kaggle/input/datasets/cuongnguyen1802/hrp4k-dataset-2/HRP4K"),
    Path("/kaggle/input/datasets/cuongnguyen1802/hrp4k-dataset-2"),
    Path("/kaggle/input/hrp4k-dataset-2/HRP4K/HRP4K"),
    Path("/kaggle/input/hrp4k-dataset-2/HRP4K"),
    Path("/kaggle/input/hrp4k-dataset-2"),
    Path("/kaggle/input/hrp4k/HRP4K"),
    Path("/kaggle/input/hrp4k"),
]


def resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    """Resolve HRP4K dataset path, checking local path first, then Kaggle input datasets before fallback."""
    if data_dir is not None:
        target = Path(data_dir).expanduser()
        if target.is_dir() and (target / "train").is_dir() and (target / "train.json").is_file():
            return target
        if target.is_dir() and (target / "HRP4K" / "train").is_dir():
            return target / "HRP4K"

    # Check default local HRP4K directory
    local_default = Path("HRP4K")
    if local_default.is_dir() and (local_default / "train").is_dir() and (local_default / "train.json").is_file():
        return local_default
    if local_default.is_dir() and (local_default / "HRP4K" / "train").is_dir():
        return local_default / "HRP4K"

    # Check Kaggle dataset input paths
    for candidate in KAGGLE_CANDIDATE_PATHS:
        if candidate.is_dir() and (candidate / "train").is_dir() and (candidate / "train.json").is_file():
            return candidate
        if candidate.is_dir() and (candidate / "HRP4K" / "train").is_dir() and (candidate / "HRP4K" / "train.json").is_file():
            return candidate / "HRP4K"

    return Path(data_dir or "HRP4K")


def ensure_dataset(data_dir: Path | str | None = None, auto_download: bool = True) -> tuple[Path, str]:
    """Ensure dataset exists: check Kaggle input first, then local, then auto-download from HF.
    
    Returns (resolved_path, source_type).
    """
    resolved = resolve_data_dir(data_dir)
    if resolved.is_dir() and (resolved / "train").is_dir() and (resolved / "train.json").is_file():
        source_type = "kaggle_input" if str(resolved).startswith("/kaggle/input") else "local"
        if source_type == "kaggle_input" and not Path("HRP4K").exists():
            try:
                os.symlink(str(resolved), "HRP4K")
            except Exception:
                pass
        return resolved, source_type

    if not auto_download:
        return resolved, "missing"

    # Auto download from Hugging Face
    repo_id = os.environ.get("HF_REPO", "Cuong2004/HRP4K")
    token = os.environ.get("HF_TOKEN")
    print(f"Dataset not found in Kaggle input or locally. Downloading from Hugging Face ({repo_id})...")
    try:
        from huggingface_hub import hf_hub_download
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename="HRP4K.zip",
            repo_type="dataset",
            local_dir=".",
            token=token,
        )
        print("Extracting HRP4K.zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        extracted = resolve_data_dir("HRP4K")
        return extracted, "huggingface"
    except Exception as exc:
        print(f"Warning: Auto-download failed: {exc}")
        return Path(data_dir or "HRP4K"), "failed"


def image_path(data_dir: Path, split: str, file_name: str) -> Path:
    resolved = resolve_data_dir(data_dir)
    return resolved / split / "images" / Path(file_name).name
