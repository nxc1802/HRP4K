import tempfile
import unittest
from pathlib import Path
from hrp4k.cli import build_parser, main
from hrp4k.phases.phase_0 import run_phase_0
from hrp4k.phases.phase_3 import run_phase_3
from hrp4k.data.views import prepare_dataset_view


class TestPipelineIntegration(unittest.TestCase):
    def test_cli_parser_subcommands(self):
        parser = build_parser()
        args = parser.parse_args(["config", "show", "--detector", "yolo11m", "--profile", "smoke"])
        self.assertEqual(args.command, "config")
        self.assertEqual(args.config_command, "show")
        self.assertEqual(args.detector, "yolo11m")
        self.assertEqual(args.profile, "smoke")

        args = parser.parse_args(["config", "validate", "--detector", "yolo11m"])
        self.assertEqual(args.config_command, "validate")

        args = parser.parse_args(["experiment", "id", "--config", "configs/base.yaml"])
        self.assertEqual(args.command, "experiment")
        self.assertEqual(args.experiment_command, "id")

    def test_phase0_on_data(self):
        data_dir = Path("HRP4K")
        if data_dir.is_dir():
            with tempfile.TemporaryDirectory() as tmpdir:
                out = Path(tmpdir)
                res = run_phase_0(data_dir, out, quality_samples=0)
                self.assertIn("integrity", res)
                self.assertIn("summary", res)
                self.assertTrue((out / "dataset_analysis_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
