from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from hrp4k.infra.upload import BackgroundHFSyncer, get_hf_credentials, load_dotenv
from hrp4k.phases.phase_1 import run_phase_1


class TestCloudSyncAndPhase1(unittest.TestCase):
    def test_load_dotenv_parsing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text(
                "# This is a comment\n"
                "HF_TOKEN=hf_test_token_123\n"
                'HF_REPO="user/my-hrp4k-repo"\n'
                "EMPTY_VAL=\n"
                "SPACED_KEY = 'spaced_val'\n",
                encoding="utf-8",
            )

            # Clear any preexisting env var for test
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HF_REPO", None)
            os.environ.pop("SPACED_KEY", None)

            loaded = load_dotenv(env_file)
            self.assertEqual(loaded.get("HF_TOKEN"), "hf_test_token_123")
            self.assertEqual(loaded.get("HF_REPO"), "user/my-hrp4k-repo")
            self.assertEqual(loaded.get("SPACED_KEY"), "spaced_val")
            self.assertEqual(os.environ.get("HF_TOKEN"), "hf_test_token_123")

    def test_get_hf_credentials_resolution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text("HF_TOKEN=hf_from_dotenv\nHF_REPO=owner/repo_from_dotenv\n", encoding="utf-8")

            with patch("hrp4k.infra.upload.load_dotenv", side_effect=lambda: load_dotenv(env_file)):
                os.environ.pop("HF_TOKEN", None)
                os.environ.pop("HF_REPO", None)

                # 1. From dotenv
                token, repo, rtype = get_hf_credentials()
                self.assertEqual(token, "hf_from_dotenv")
                self.assertEqual(repo, "owner/repo_from_dotenv")
                self.assertEqual(rtype, "dataset")

                # 2. Explicit overrides
                token, repo, rtype = get_hf_credentials(token="hf_explicit", repo_id="custom/repo", repo_type="model")
                self.assertEqual(token, "hf_explicit")
                self.assertEqual(repo, "custom/repo")
                self.assertEqual(rtype, "model")

    def test_syncer_disabled_when_no_token(self):
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("", encoding="utf-8")
            with patch("hrp4k.infra.upload.load_dotenv", return_value={}):
                syncer = BackgroundHFSyncer(token=None, enabled=True)
                self.assertFalse(syncer.enabled)
                # Ensure calls do not raise error
                syncer.sync_epoch(1, Path(tmp))
                syncer.wait_until_done(timeout=1.0)
                syncer.shutdown()

    @patch("huggingface_hub.HfApi")
    def test_syncer_background_upload_flow(self, mock_hf_api_class):
        mock_api_instance = MagicMock()
        mock_hf_api_class.return_value = mock_api_instance

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            weights_dir = tmp_path / "weights"
            weights_dir.mkdir()
            best_pt = weights_dir / "best.pt"
            best_pt.write_text("dummy best", encoding="utf-8")
            last_pt = weights_dir / "last.pt"
            last_pt.write_text("dummy last", encoding="utf-8")
            results_csv = tmp_path / "results.csv"
            results_csv.write_text("epoch,loss\n1,0.5\n", encoding="utf-8")

            syncer = BackgroundHFSyncer(
                repo_id="test/hrp4k",
                token="hf_mock_token",
                path_in_repo="checkpoints/yolo11m",
                enabled=True,
            )
            self.assertTrue(syncer.enabled)

            syncer.sync_epoch(
                epoch=1,
                weights_dir=weights_dir,
                extra_files=[results_csv],
            )
            syncer.wait_until_done(timeout=5.0)
            syncer.shutdown(wait=True)

            # Verify that mock_api_instance.upload_file was called for files
            self.assertTrue(mock_api_instance.upload_file.called)
            uploaded_paths = [call.kwargs.get("path_in_repo") for call in mock_api_instance.upload_file.call_args_list]
            self.assertIn("checkpoints/yolo11m/best.pt", uploaded_paths)
            self.assertIn("checkpoints/yolo11m/last.pt", uploaded_paths)
            self.assertIn("checkpoints/yolo11m/results.csv", uploaded_paths)


if __name__ == "__main__":
    unittest.main()
