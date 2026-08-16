from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def upload_to_hf(
    repo_id: str,
    local_path: Path | str,
    token: str,
    repo_type: str = "dataset",
    path_in_repo: str | None = None,
    commit_message: str | None = None,
) -> dict[str, Any]:
    """Upload checkpoints and artifacts folder or file to Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    source = Path(local_path).expanduser()

    if not source.exists():
        raise FileNotFoundError(f"Path does not exist: {source}")

    commit_msg = commit_message or f"Upload checkpoints and artifacts from {source.name}"

    if source.is_dir():
        print(f"Uploading directory '{source}' to Hugging Face repository '{repo_id}' ({repo_type})...")
        api.upload_folder(
            folder_path=str(source),
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=path_in_repo or source.name,
            commit_message=commit_msg,
        )
    else:
        print(f"Uploading file '{source}' to Hugging Face repository '{repo_id}' ({repo_type})...")
        api.upload_file(
            path_or_fileobj=str(source),
            path_in_repo=path_in_repo or source.name,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_msg,
        )

    return {
        "status": "success",
        "repo_id": repo_id,
        "repo_type": repo_type,
        "source": str(source),
        "path_in_repo": path_in_repo or source.name,
    }
