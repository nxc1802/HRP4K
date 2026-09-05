#!/usr/bin/env python3
"""Runner for Focal Loss Only Optimization Experiment.

Target Assignment: 1x1 (Baseline)
Classification Loss: Sigmoid Focal Loss (alpha=0.25, gamma=2.0)
Scale Weights: Uniform (1.0, 1.0, 1.0, 1.0)
Epochs: 30
Patience: 5
Inference Setup: Top-K=300, P2 Conf=0.001, Fusion NMS IoU=0.6
"""

import os
import sys
import json
import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR / "src"))

OUTPUT_DIR = BASE_DIR / "outputs" / "optimization_pipeline" / "phase3_focal_loss_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "focal_experiment.log"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
HEARTBEAT_FILE = OUTPUT_DIR / "heartbeat.json"

stop_heartbeat = threading.Event()


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def heartbeat_worker():
    while not stop_heartbeat.is_set():
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "status": "running",
                "pid": os.getpid(),
            }
            HEARTBEAT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
        time.sleep(20)


def update_progress(phase: str, current_task: str, status: str = "running"):
    data = {
        "status": status,
        "phase": phase,
        "current_task": current_task,
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_cmd_logged(cmd: list[str], task_name: str) -> int:
    log(f"--- STARTING: {task_name} ---")
    log(f"Command: {' '.join(cmd)}")
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BASE_DIR / 'src'}:{env.get('PYTHONPATH', '')}"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(BASE_DIR),
        env=env,
    )

    for line in proc.stdout:
        line_clean = line.rstrip()
        if line_clean:
            log(f"  {line_clean}")

    proc.wait()
    duration = time.time() - t0
    log(f"--- FINISHED: {task_name} (Exit code: {proc.returncode}, Duration: {duration:.1f}s) ---\n")
    return proc.returncode


def main() -> int:
    log("=" * 80)
    log("STARTING FOCAL LOSS ONLY EXPERIMENT (Epoch 30, Patience 5)")
    log("=" * 80)

    hb_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    hb_thread.start()

    data_yaml = BASE_DIR / "outputs" / "full_dataset" / "dataset.yaml"
    if not data_yaml.is_file():
        data_yaml = BASE_DIR / "HRP4K" / "data.yaml"

    base_weights = BASE_DIR / "outputs" / "experiments" / "rtdetr-l-resolution-2k" / "weights" / "best.pt"
    device = "0"

    try:
        # STEP 1: Train Focal Loss Only
        update_progress("Training", "Focal Loss P2-Only Training (Epochs: 30, Patience: 5)")
        train_cmd = [
            sys.executable, "-u",
            str(BASE_DIR / "train_p2_optimization.py"),
            "--phase", "3",
            "--name", "focal_loss_only",
            "--target-assignment", "1x1",
            "--cls-loss", "focal",
            "--epochs", "30",
            "--patience", "5",
            "--batch", "16",
            "--weights", str(base_weights),
            "--data", str(data_yaml),
            "--device", device,
            "--output", str(OUTPUT_DIR),
            "--topk", "300",
            "--p2-conf", "0.001",
            "--fusion-iou", "0.6",
            "--allow-full",
            "--hf-upload",
        ]

        ret_train = run_cmd_logged(train_cmd, "Step 1: Train Focal Loss Only")
        if ret_train != 0:
            log(f"[Error] Training failed with exit code {ret_train}")
            update_progress("Failed", f"Training exited with code {ret_train}", status="failed")
            return ret_train

        best_p2_path = OUTPUT_DIR / "weights" / "best_p2.pt"
        if not best_p2_path.is_file():
            log(f"[Error] Best checkpoint not found at {best_p2_path}")
            return 1

        # STEP 2: Comprehensive Head-to-Head Evaluation
        update_progress("Evaluation", "Running Full Comparison Benchmark on 900 Test Images")
        eval_output = OUTPUT_DIR / "evaluation"
        eval_output.mkdir(parents=True, exist_ok=True)

        eval_cmd = [
            sys.executable, "-u",
            str(BASE_DIR / "eval_comparison.py"),
            "--checkpoint", str(best_p2_path),
            "--weights", str(base_weights),
            "--data", str(data_yaml.parent if data_yaml.is_file() else data_yaml),
            "--device", device,
            "--output", str(eval_output),
            "--topk", "300",
            "--p2-conf", "0.001",
            "--fusion-iou", "0.6",
            "--hf-upload",
        ]

        ret_eval = run_cmd_logged(eval_cmd, "Step 2: Head-to-Head Comparison Benchmark")
        if ret_eval != 0:
            log(f"[Warning] Evaluation exited with code {ret_eval}")

        update_progress("Complete", "Focal Loss Only Experiment Finished", status="completed")
        log("=" * 80)
        log("FOCAL LOSS ONLY EXPERIMENT COMPLETED SUCCESSFULLY")
        log("=" * 80)

    except Exception as e:
        log(f"[FATAL ERROR] {e}")
        update_progress("Failed", str(e), status="failed")
        return 1
    finally:
        stop_heartbeat.set()

    return 0


if __name__ == "__main__":
    sys.exit(main())
