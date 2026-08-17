from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any


def load_dotenv(dotenv_path: Path | str | None = None) -> dict[str, str]:
    """Parse key-value pairs from a .env file and set them into os.environ if not already present."""
    env_vars: dict[str, str] = {}
    candidate_paths = []

    if dotenv_path is not None:
        candidate_paths.append(Path(dotenv_path).expanduser())
    else:
        cwd = Path.cwd()
        candidate_paths.append(cwd / ".env")
        # Check parents up to workspace root
        for parent in cwd.parents:
            candidate_paths.append(parent / ".env")
            if (parent / ".git").exists():
                break

    target_file = None
    for p in candidate_paths:
        if p.is_file():
            target_file = p
            break

    if target_file is None:
        return env_vars

    try:
        content = target_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Strip single or double quotes if matching
                if len(val) >= 2 and ((val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'"))):
                    val = val[1:-1]
                env_vars[key] = val
                if key not in os.environ:
                    os.environ[key] = val
    except Exception as exc:
        print(f"[Warning] Failed reading .env from {target_file}: {exc}")

    return env_vars


def get_hf_credentials(
    token: str | None = None,
    repo_id: str | None = None,
    repo_type: str = "dataset",
) -> tuple[str | None, str | None, str]:
    """Resolve Hugging Face Hub token, repository ID, and repo type from args, environment, and .env."""
    load_dotenv()
    resolved_token = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    resolved_repo = (
        repo_id
        or os.environ.get("HF_REPO")
        or os.environ.get("HUGGINGFACE_REPO")
        or "Cuong2004/HRP4K"
    )
    resolved_type = os.environ.get("HF_REPO_TYPE") or repo_type
    return resolved_token, resolved_repo, resolved_type


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
            ignore_patterns=["full_dataset/**", "local_dataset/**", "smoke/**", "*.cache", "*.tmp", "__pycache__/**"],
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


class BackgroundHFSyncer:
    """Asynchronous background Hugging Face checkpoint syncer running on a dedicated worker thread.
    
    Ensures model checkpoints ('best.pt', 'last.pt') and metrics ('results.csv', 'args.yaml')
    are periodically synced to Hugging Face Cloud without blocking GPU training iterations.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        token: str | None = None,
        repo_type: str = "dataset",
        path_in_repo: str | None = None,
        enabled: bool = True,
    ):
        self.token, self.repo_id, self.repo_type = get_hf_credentials(token, repo_id, repo_type)
        self.default_path_in_repo = path_in_repo
        self.enabled = bool(enabled and self.token and self.repo_id)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stopped = False

        if self.enabled:
            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="HFSyncerWorker")
            self._thread.start()
            print(f"[Cloud Sync] Initialized background Hugging Face sync -> '{self.repo_id}' ({self.repo_type})")
        else:
            if enabled and not self.token:
                print("[Cloud Sync] Skipped background sync: HF_TOKEN not found in environment or .env")

    def _worker_loop(self) -> None:
        """Background thread worker pulling upload tasks from queue."""
        from huggingface_hub import HfApi

        api = HfApi(token=self.token)

        while not self._stopped:
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:  # Shutdown sentinel
                self._queue.task_done()
                break

            try:
                epoch = task.get("epoch", 0)
                files_to_upload = task.get("files", [])
                target_folder = task.get("path_in_repo") or self.default_path_in_repo or "checkpoints"

                for file_path, file_target_name in files_to_upload:
                    p = Path(file_path)
                    if not p.is_file():
                        continue
                    dest_path = f"{target_folder}/{file_target_name}" if target_folder else file_target_name
                    dest_path = dest_path.replace("\\", "/").strip("/")

                    api.upload_file(
                        path_or_fileobj=str(p),
                        path_in_repo=dest_path,
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        commit_message=f"Auto-sync epoch {epoch}: {file_target_name}",
                    )

                print(f"[Cloud Sync] Epoch {epoch}: Successfully uploaded {len(files_to_upload)} files to '{self.repo_id}/{target_folder}'")
            except Exception as exc:
                print(f"[Cloud Sync Warning] Background upload failed at epoch {task.get('epoch')}: {exc} (Training continues uninterrupted)")
            finally:
                self._queue.task_done()

    def sync_epoch(
        self,
        epoch: int,
        weights_dir: Path | str,
        extra_files: list[Path | str] | None = None,
        path_in_repo: str | None = None,
    ) -> None:
        """Enqueue an asynchronous checkpoint upload task for the given epoch."""
        if not self.enabled:
            return

        w_dir = Path(weights_dir)
        files_to_upload: list[tuple[Path, str]] = []

        best_pt = w_dir / "best.pt"
        last_pt = w_dir / "last.pt"

        if best_pt.is_file():
            files_to_upload.append((best_pt, "best.pt"))
        if last_pt.is_file():
            files_to_upload.append((last_pt, "last.pt"))

        if extra_files:
            for ef in extra_files:
                ef_path = Path(ef)
                if ef_path.is_file():
                    files_to_upload.append((ef_path, ef_path.name))

        if not files_to_upload:
            return

        task = {
            "epoch": epoch,
            "files": files_to_upload,
            "path_in_repo": path_in_repo or self.default_path_in_repo,
            "timestamp": time.time(),
        }
        self._queue.put(task)

    def wait_until_done(self, timeout: float = 60.0) -> None:
        """Block until all queued upload tasks are finished or timeout expires."""
        if not self.enabled:
            return
        try:
            start = time.time()
            while not self._queue.empty():
                if time.time() - start > timeout:
                    print(f"[Cloud Sync Warning] Timeout ({timeout}s) waiting for background uploads to finish.")
                    break
                time.sleep(0.5)
            self._queue.join()
        except Exception:
            pass

    def shutdown(self, wait: bool = True, timeout: float = 60.0) -> None:
        """Gracefully stop the background worker."""
        if not self.enabled:
            return
        if wait:
            self.wait_until_done(timeout=timeout)
        self._stopped = True
        self._queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
