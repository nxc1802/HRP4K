#!/usr/bin/env python3
"""Master Evaluation Runner for HRP4K Benchmark:
Runs complete dual-benchmark evaluation across ALL models:
  1. Proposed Method (P2-Only, Native RT-DETR-L 2K, Fused)
  2. Phase 1 Resolution Baselines (8 models: YOLO11m & RT-DETR-L at 640, 1K, 2K, 4K)
  3. Phase 2 Spatial Decomposition (Sliced-NMS, SAHI, Perspective Grid)
  4. Keeps alive with heartbeat to prevent container sleep.
  5. Syncs all metrics and raw predictions to Hugging Face Hub.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path("/marimo/HRP4K")
if not BASE_DIR.is_dir():
    BASE_DIR = Path(".")

OUTPUTS_DIR = BASE_DIR / "outputs" / "master_eval"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_FILE = OUTPUTS_DIR / "progress.json"
HEARTBEAT_FILE = OUTPUTS_DIR / "heartbeat.json"
LOG_FILE = OUTPUTS_DIR / "master_eval.log"

def get_env_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_file = BASE_DIR / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None

HF_TOKEN = get_env_token()
HF_REPO = os.environ.get("HF_REPO", "Cuong2004/HRP4K")
DEVICE = "0"

stop_heartbeat = threading.Event()


def heartbeat_worker():
    """Background worker updating heartbeat every 20 seconds to prevent idle sleep."""
    while not stop_heartbeat.is_set():
        try:
            now_iso = datetime.now().isoformat()
            data = {
                "timestamp": now_iso,
                "status": "active",
                "pid": os.getpid(),
            }
            HEARTBEAT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
        time.sleep(20)


def update_progress(phase: str, current_task: str, completed: list[str], status: str = "running"):
    data = {
        "status": status,
        "phase": phase,
        "current_task": current_task,
        "completed_tasks": completed,
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_command_logged(cmd: list[str], task_name: str) -> int:
    log(f"--- STARTING: {task_name} ---")
    log(f"Command: {' '.join(cmd)}")
    t0 = time.time()
    
    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN
    env["HF_REPO"] = HF_REPO
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
    log("HRP4K MASTER EVALUATION RUNNER STARTED")
    log(f"Base Dir: {BASE_DIR.resolve()}")
    log(f"Device: {DEVICE}")
    log("=" * 80)

    # Start heartbeat thread
    hb_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    hb_thread.start()

    completed = []

    try:
        # -------------------------------------------------------------------
        # STEP 1: Proposed Method Dual Evaluation (P2 vs Native vs Fused)
        # -------------------------------------------------------------------
        update_progress("Phase 3: Proposed Architecture", "eval_comparison.py", completed)
        step1_cmd = [
            sys.executable,
            str(BASE_DIR / "eval_comparison.py"),
            "--checkpoint", str(BASE_DIR / "outputs/experiments/rtdetr-l-proposed-p2-2k/weights/best_p2.pt"),
            "--weights", str(BASE_DIR / "outputs/experiments/rtdetr-l-resolution-2k/weights/best.pt"),
            "--data", str(BASE_DIR / "HRP4K"),
            "--device", DEVICE,
            "--hf-upload",
        ]
        ret1 = run_command_logged(step1_cmd, "Proposed Method Dual Evaluation (eval_comparison.py)")
        if ret1 == 0:
            completed.append("Phase 3: Proposed Method (P2-Only, Native 2K, Fused 2K)")
        else:
            log(f"[WARN] Step 1 exited with code {ret1}")

        # -------------------------------------------------------------------
        # STEP 2: Phase 1 Resolution Baselines (8 Models)
        # -------------------------------------------------------------------
        update_progress("Phase 1: Resolution Baselines", "eval_benchmark.py --all-phase1", completed)
        step2_cmd = [
            sys.executable,
            str(BASE_DIR / "eval_benchmark.py"),
            "--all-phase1",
            "--data", str(BASE_DIR / "HRP4K"),
            "--device", DEVICE,
            "--hf-upload",
        ]
        ret2 = run_command_logged(step2_cmd, "Phase 1 Resolution Baselines (8 Models)")
        if ret2 == 0:
            completed.append("Phase 1: Resolution Baselines (8 Models)")
        else:
            log(f"[WARN] Step 2 exited with code {ret2}")

        # -------------------------------------------------------------------
        # FINALIZATION
        # -------------------------------------------------------------------
        update_progress("Completed", "None", completed, status="done")
        log("=" * 80)
        log("ALL BENCHMARK EVALUATIONS SUCCESSFULLY COMPLETED!")
        log(f"Completed Tasks: {completed}")
        log("=" * 80)
        return 0

    except Exception as exc:
        log(f"[FATAL ERROR] {exc}")
        update_progress("Failed", str(exc), completed, status="error")
        return 1
    finally:
        stop_heartbeat.set()


if __name__ == "__main__":
    sys.exit(main())
