"""Hugging Face persistent experiment storage.

Implements the experiment folder structure on HF:
    experiments/<experiment_id>/
    ├── manifest.json
    ├── config.json
    ├── environment.json
    ├── training/
    │   ├── history.jsonl
    │   ├── results.csv
    │   └── epochs/epoch-001/ ...
    ├── checkpoints/last.pt, best.pt
    ├── validation/metrics.json
    └── test/predictions.json, metrics.json
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .upload import get_hf_credentials, is_placeholder_token


@dataclass
class CheckpointInfo:
    epoch: int
    path: str
    best_metric: float | None = None
    timestamp: str | None = None


@dataclass
class ExperimentState:
    exists: bool = False
    latest_epoch: int = 0
    best_epoch: int = 0
    best_metric: float = 0.0
    checkpoint_path: str | None = None
    status: str = "not_found"


class ExperimentStorage:
    """Manages experiment artifacts on Hugging Face Hub."""

    def __init__(
        self,
        experiment_id: str,
        repo_id: str | None = None,
        token: str | None = None,
        repo_type: str = "dataset",
    ):
        self.experiment_id = experiment_id
        self.token, self.repo_id, self.repo_type = get_hf_credentials(token, repo_id, repo_type)
        self.enabled = bool(self.token and self.repo_id)
        self._base_path = f"experiments/{experiment_id}"

    def check_experiment_exists(self) -> ExperimentState:
        """Check if experiment already exists on HF and return its state."""
        if not self.enabled:
            return ExperimentState(exists=False, status="hf_disabled")

        try:
            from huggingface_hub import list_repo_files
            repo_files = list_repo_files(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                token=self.token,
            )
            # Check for manifest
            manifest_path = f"{self._base_path}/manifest.json"
            experiment_files = [f for f in repo_files if f.startswith(self._base_path)]

            if not experiment_files:
                return ExperimentState(exists=False, status="not_found")

            # Find latest epoch
            epoch_dirs = [f for f in experiment_files if "/training/epochs/epoch-" in f]
            latest_epoch = 0
            for f in epoch_dirs:
                try:
                    epoch_num = int(f.split("epoch-")[1].split("/")[0])
                    latest_epoch = max(latest_epoch, epoch_num)
                except (IndexError, ValueError):
                    continue

            # Try to read manifest for best metric info
            best_metric = 0.0
            best_epoch = 0
            if manifest_path in experiment_files:
                try:
                    from huggingface_hub import hf_hub_download
                    local = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=manifest_path,
                        repo_type=self.repo_type,
                        token=self.token,
                    )
                    manifest = json.loads(Path(local).read_text())
                    best_metric = manifest.get("best_metric", 0.0)
                    best_epoch = manifest.get("best_epoch", 0)
                except Exception:
                    pass

            # Determine checkpoint path
            checkpoint_path = None
            for candidate in [
                f"{self._base_path}/checkpoints/last.pt",
                f"{self._base_path}/checkpoints/best.pt",
                f"{self._base_path}/checkpoints/best_p2.pt",
                f"{self._base_path}/weights/best_p2.pt",
                f"{self._base_path}/best_p2.pt",
                f"{self._base_path}/training/epochs/epoch-{latest_epoch:03d}/last.pt",
            ]:
                if candidate in experiment_files:
                    checkpoint_path = candidate
                    break

            return ExperimentState(
                exists=True,
                latest_epoch=latest_epoch,
                best_epoch=best_epoch,
                best_metric=best_metric,
                checkpoint_path=checkpoint_path,
                status="resumable" if checkpoint_path else "exists_no_checkpoint",
            )
        except Exception as exc:
            print(f"[HF Storage] Warning: Could not check experiment state: {exc}")
            return ExperimentState(exists=False, status=f"error: {exc}")

    def download_checkpoint(self, epoch: int | None = None) -> Path | None:
        """Download the latest or specified epoch checkpoint from HF."""
        if not self.enabled:
            return None

        try:
            from huggingface_hub import hf_hub_download

            # Try specific epoch first, then last.pt / best.pt / best_p2.pt
            candidates = []
            if epoch is not None:
                candidates.append(f"{self._base_path}/training/epochs/epoch-{epoch:03d}/last.pt")
            candidates.extend([
                f"{self._base_path}/checkpoints/last.pt",
                f"{self._base_path}/checkpoints/best.pt",
                f"{self._base_path}/checkpoints/best_p2.pt",
                f"{self._base_path}/weights/best_p2.pt",
                f"{self._base_path}/best_p2.pt",
            ])

            for candidate in candidates:
                try:
                    local = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=candidate,
                        repo_type=self.repo_type,
                        token=self.token,
                    )
                    print(f"[HF Storage] Downloaded checkpoint: {candidate}")
                    return Path(local)
                except Exception:
                    continue

            print("[HF Storage] No checkpoint found on HF for this experiment.")
            return None
        except Exception as exc:
            print(f"[HF Storage] Warning: Could not download checkpoint: {exc}")
            return None

    def upload_file(self, local_path: Path, remote_name: str, commit_message: str = "") -> bool:
        """Upload a single file to the experiment folder on HF. Never crashes."""
        if not self.enabled:
            return False
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.token)
            remote_path = f"{self._base_path}/{remote_name}"
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                commit_message=commit_message or f"Upload {remote_name}",
            )
            return True
        except Exception as exc:
            print(f"[HF Storage] Warning: Upload failed for {remote_name}: {exc}")
            return False

    def upload_epoch(
        self,
        epoch: int,
        weights_dir: Path,
        extra_files: list[Path] | None = None,
    ) -> dict[str, Any]:
        """Upload all artifacts for a training epoch. Never crashes training."""
        sync_status = {"epoch": epoch, "timestamp": time.time(), "uploaded": [], "failed": []}
        if not self.enabled:
            sync_status["status"] = "disabled"
            return sync_status

        epoch_prefix = f"training/epochs/epoch-{epoch:03d}"

        # Upload weight files
        if weights_dir.is_dir():
            for pt_file in sorted(weights_dir.glob("*.pt")):
                if pt_file.is_file():
                    # Upload to epoch folder (historical)
                    ok = self.upload_file(pt_file, f"{epoch_prefix}/{pt_file.name}",
                                         f"Epoch {epoch}: {pt_file.name}")
                    # Also update convenience pointers
                    self.upload_file(pt_file, f"checkpoints/{pt_file.name}",
                                    f"Epoch {epoch}: update {pt_file.name}")
                    (sync_status["uploaded"] if ok else sync_status["failed"]).append(pt_file.name)

        # Upload extra files (metrics, results, etc.)
        if extra_files:
            for ef in extra_files:
                if ef.is_file():
                    ok = self.upload_file(ef, f"{epoch_prefix}/{ef.name}",
                                         f"Epoch {epoch}: {ef.name}")
                    (sync_status["uploaded"] if ok else sync_status["failed"]).append(ef.name)

        sync_status["status"] = "complete" if not sync_status["failed"] else "partial"
        return sync_status

    def upload_manifest(self, manifest: dict[str, Any]) -> bool:
        """Upload experiment manifest.json."""
        if not self.enabled:
            return False
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
                tmp_path = Path(f.name)
            result = self.upload_file(tmp_path, "manifest.json", "Update experiment manifest")
            tmp_path.unlink(missing_ok=True)
            return result
        except Exception as exc:
            print(f"[HF Storage] Warning: Could not upload manifest: {exc}")
            return False

    def upload_config(self, config: dict[str, Any]) -> bool:
        """Upload experiment config.json."""
        if not self.enabled:
            return False
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                tmp_path = Path(f.name)
            result = self.upload_file(tmp_path, "config.json", "Upload experiment config")
            tmp_path.unlink(missing_ok=True)
            return result
        except Exception as exc:
            print(f"[HF Storage] Warning: Could not upload config: {exc}")
            return False

    def upload_final_results(
        self,
        val_metrics_path: Path | None = None,
        test_metrics_path: Path | None = None,
        test_predictions_path: Path | None = None,
    ) -> dict[str, bool]:
        """Upload final validation/test results."""
        results = {}
        if val_metrics_path and val_metrics_path.is_file():
            results["validation/metrics.json"] = self.upload_file(
                val_metrics_path, "validation/metrics.json", "Upload validation metrics"
            )
        if test_metrics_path and test_metrics_path.is_file():
            results["test/metrics.json"] = self.upload_file(
                test_metrics_path, "test/metrics.json", "Upload test metrics"
            )
        if test_predictions_path and test_predictions_path.is_file():
            results["test/predictions.json"] = self.upload_file(
                test_predictions_path, "test/predictions.json", "Upload test predictions"
            )
        return results
