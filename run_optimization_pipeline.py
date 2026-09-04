#!/usr/bin/env python3
"""Master Optimization Pipeline Runner for Proposed Method (Phases 1 - 6).

Executes the end-to-end optimization workflow defined in Big Plan.md:
  Phase 0: Baseline Verification (Recorded Native & Fused metrics)
  Phase 1: Zero-Compute Inference-Time Optimization Sweep (Top-K, P2 threshold, NMS IoU)
  Phase 2: Multi-Positive Target Assignment (3x3 vs 1x1)
  Phase 3: Focal Loss / Quality Focal Loss (QFL)
  Phase 4: Scale-Aware Loss Weighting (Ultra-fine prioritization)
  Phase 5: Build Best P2 Combination
  Phase 6: Canonical Comprehensive Evaluation & Decision Gate Analysis

Features:
  - Background Heartbeat thread preventing server / container sleep.
  - Granular control: execute all phases or select specific phases (e.g. --phases 1,6).
  - Dry-run and smoke-test modes for local syntax and runtime verification.
  - Cloud synchronization to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Add src/ to sys.path
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR / "src"))

OUTPUTS_DIR = BASE_DIR / "outputs" / "optimization_pipeline"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_FILE = OUTPUTS_DIR / "progress.json"
HEARTBEAT_FILE = OUTPUTS_DIR / "heartbeat.json"
LOG_FILE = OUTPUTS_DIR / "pipeline.log"

stop_heartbeat = threading.Event()


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


def heartbeat_worker():
    """Background heartbeat updating every 20s to prevent container idle sleep."""
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


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def update_progress(phase: str, current_task: str, completed: list[str], status: str = "running"):
    data = {
        "status": status,
        "phase": phase,
        "current_task": current_task,
        "completed_tasks": completed,
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_cmd_logged(cmd: list[str], task_name: str) -> int:
    log(f"--- STARTING: {task_name} ---")
    log(f"Command: {' '.join(cmd)}")
    t0 = time.time()

    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN or ""
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
    parser = argparse.ArgumentParser(description="HRP4K Proposed Method Optimization Pipeline (Phases 1-6)")
    parser.add_argument(
        "--phases",
        default="1,2,3,4,5,6",
        help="Comma-separated list of phases to execute (e.g. '1', '1,6', or '1,2,3,4,5,6')",
    )
    parser.add_argument(
        "--weights",
        default="outputs/experiments/rtdetr-l-resolution-2k/weights/best.pt",
        help="Base fine-tuned RT-DETR model checkpoint",
    )
    parser.add_argument(
        "--p2-base",
        default="outputs/experiments/rtdetr-l-proposed-p2-2k/weights/best_p2.pt",
        help="Original P2 head checkpoint for Phase 1 sweeps & baseline comparison",
    )
    parser.add_argument(
        "--data",
        default="HRP4K",
        help="Dataset path",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index or 'cpu'",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke verification (1 epoch, minimal iterations)",
    )
    parser.add_argument(
        "--allow-full",
        action="store_true",
        help="Required flag to execute full GPU training runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands and execution schedule without running",
    )
    parser.add_argument(
        "--hf-upload",
        action="store_true",
        help="Upload checkpoints and metrics to Hugging Face",
    )

    args = parser.parse_args()
    selected_phases = set()
    for p in args.phases.split(","):
        p_clean = p.strip()
        if p_clean.isdigit():
            selected_phases.add(int(p_clean))

    log("=" * 80)
    log("HRP4K PROPOSED METHOD OPTIMIZATION PIPELINE STARTED")
    log(f"Selected Phases: {sorted(list(selected_phases))}")
    log(f"Base Weights:    {args.weights}")
    log(f"P2 Base Weights: {args.p2_base}")
    log(f"Device:          {args.device}")
    log(f"Smoke Mode:      {args.smoke}")
    log(f"Allow Full:      {args.allow_full}")
    log("=" * 80)

    if args.dry_run:
        log("\n[DRY RUN] Planned Execution Schedule:")
        if 1 in selected_phases:
            log("  - Phase 1: python3 sweep_p2_inference.py --checkpoint ... --device ...")
        if 2 in selected_phases:
            log("  - Phase 2: python3 train_p2_optimization.py --phase 2 --target-assignment 3x3 ...")
        if 3 in selected_phases:
            log("  - Phase 3: python3 train_p2_optimization.py --phase 3 --target-assignment 3x3 --cls-loss qfl ...")
        if 4 in selected_phases:
            log("  - Phase 4: python3 train_p2_optimization.py --phase 4 --scale-weights 3.0,2.0,1.0,0.5 ...")
        if 5 in selected_phases:
            log("  - Phase 5: python3 train_p2_optimization.py --phase 5 (best combination) ...")
        if 6 in selected_phases:
            log("  - Phase 6: python3 eval_comparison.py --checkpoint ... --device ... (Decision Gate)")
        return 0

    # Start Heartbeat
    hb_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    hb_thread.start()

    completed = []

    # Current winning configuration defaults
    best_inf_topk = 300
    best_inf_conf = 0.001
    best_inf_iou = 0.5
    current_p2_ckpt = args.p2_base

    try:
        # ===================================================================
        # PHASE 1: Zero-Compute Inference-Time Optimization Sweep
        # ===================================================================
        if 1 in selected_phases:
            update_progress("Phase 1", "Inference Optimization Sweep", completed)
            cmd_p1 = [
                sys.executable,
                str(BASE_DIR / "sweep_p2_inference.py"),
                "--checkpoint", str(args.p2_base),
                "--weights", str(args.weights),
                "--data", str(args.data),
                "--device", str(args.device),
                "--output", str(OUTPUTS_DIR / "phase1_inference_sweep"),
            ]
            if args.smoke:
                cmd_p1.extend(["--num-images", "10", "--topk", "300,500", "--conf", "0.001,0.01", "--iou", "0.5"])
            ret_p1 = run_cmd_logged(cmd_p1, "Phase 1: Inference Sweep")
            if ret_p1 == 0:
                completed.append("Phase 1: Inference Sweep")
                best_inf_path = OUTPUTS_DIR / "phase1_inference_sweep" / "best_inference_config.json"
                if best_inf_path.is_file():
                    with open(best_inf_path, "r", encoding="utf-8") as f:
                        best_inf = json.load(f)
                    best_inf_topk = best_inf.get("topk", 300)
                    best_inf_conf = best_inf.get("p2_conf_threshold", 0.001)
                    best_inf_iou = best_inf.get("fusion_iou_threshold", 0.5)
                    log(f"[Phase 1 Result] Winner: Top-K={best_inf_topk}, Conf={best_inf_conf}, IoU={best_inf_iou}")
            else:
                log(f"[Phase 1 Error] Exited with code {ret_p1}")

        # ===================================================================
        # PHASE 2: Multi-Positive Target Assignment (3x3 center region)
        # ===================================================================
        if 2 in selected_phases:
            update_progress("Phase 2", "Multi-Positive 3x3 Assignment Training", completed)
            cmd_p2 = [
                sys.executable,
                str(BASE_DIR / "train_p2_optimization.py"),
                "--phase", "2",
                "--target-assignment", "3x3",
                "--weights", str(args.weights),
                "--data", str(Path(args.data) / "data.yaml"),
                "--device", str(args.device),
                "--output", str(OUTPUTS_DIR / "phase2_multi_positive_3x3"),
                "--topk", str(best_inf_topk),
                "--p2-conf", str(best_inf_conf),
                "--fusion-iou", str(best_inf_iou),
            ]
            if args.smoke:
                cmd_p2.append("--smoke")
            elif args.allow_full:
                cmd_p2.append("--allow-full")
            if args.hf_upload:
                cmd_p2.append("--hf-upload")

            ret_p2 = run_cmd_logged(cmd_p2, "Phase 2: Multi-Positive 3x3 Training")
            if ret_p2 == 0:
                completed.append("Phase 2: Multi-Positive 3x3")
                p2_best = OUTPUTS_DIR / "phase2_multi_positive_3x3" / "weights" / "best_p2.pt"
                if p2_best.is_file():
                    current_p2_ckpt = str(p2_best)
            else:
                log(f"[Phase 2 Error] Exited with code {ret_p2}")

        # ===================================================================
        # PHASE 3: Classification Loss Function (Quality Focal Loss)
        # ===================================================================
        if 3 in selected_phases:
            update_progress("Phase 3", "Quality Focal Loss (QFL) Training", completed)
            cmd_p3 = [
                sys.executable,
                str(BASE_DIR / "train_p2_optimization.py"),
                "--phase", "3",
                "--target-assignment", "3x3",
                "--cls-loss", "qfl",
                "--weights", str(args.weights),
                "--data", str(Path(args.data) / "data.yaml"),
                "--device", str(args.device),
                "--output", str(OUTPUTS_DIR / "phase3_qfl_loss"),
                "--topk", str(best_inf_topk),
                "--p2-conf", str(best_inf_conf),
                "--fusion-iou", str(best_inf_iou),
            ]
            if args.smoke:
                cmd_p3.append("--smoke")
            elif args.allow_full:
                cmd_p3.append("--allow-full")
            if args.hf_upload:
                cmd_p3.append("--hf-upload")

            ret_p3 = run_cmd_logged(cmd_p3, "Phase 3: Quality Focal Loss Training")
            if ret_p3 == 0:
                completed.append("Phase 3: Quality Focal Loss")
                p3_best = OUTPUTS_DIR / "phase3_qfl_loss" / "weights" / "best_p2.pt"
                if p3_best.is_file():
                    current_p2_ckpt = str(p3_best)
            else:
                log(f"[Phase 3 Error] Exited with code {ret_p3}")

        # ===================================================================
        # PHASE 4: Scale-Aware Loss Weighting
        # ===================================================================
        if 4 in selected_phases:
            update_progress("Phase 4", "Scale-Aware Loss Weighting Training", completed)
            cmd_p4 = [
                sys.executable,
                str(BASE_DIR / "train_p2_optimization.py"),
                "--phase", "4",
                "--target-assignment", "3x3",
                "--cls-loss", "qfl",
                "--scale-weights", "3.0,2.0,1.0,0.5",
                "--weights", str(args.weights),
                "--data", str(Path(args.data) / "data.yaml"),
                "--device", str(args.device),
                "--output", str(OUTPUTS_DIR / "phase4_scale_aware"),
                "--topk", str(best_inf_topk),
                "--p2-conf", str(best_inf_conf),
                "--fusion-iou", str(best_inf_iou),
            ]
            if args.smoke:
                cmd_p4.append("--smoke")
            elif args.allow_full:
                cmd_p4.append("--allow-full")
            if args.hf_upload:
                cmd_p4.append("--hf-upload")

            ret_p4 = run_cmd_logged(cmd_p4, "Phase 4: Scale-Aware Weighting Training")
            if ret_p4 == 0:
                completed.append("Phase 4: Scale-Aware Weighting")
                p4_best = OUTPUTS_DIR / "phase4_scale_aware" / "weights" / "best_p2.pt"
                if p4_best.is_file():
                    current_p2_ckpt = str(p4_best)
            else:
                log(f"[Phase 4 Error] Exited with code {ret_p4}")

        # ===================================================================
        # PHASE 5: Build Best P2 Combination Run
        # ===================================================================
        if 5 in selected_phases:
            update_progress("Phase 5", "Best P2 Combination Training", completed)
            cmd_p5 = [
                sys.executable,
                str(BASE_DIR / "train_p2_optimization.py"),
                "--phase", "5",
                "--name", "best_p2_combination",
                "--target-assignment", "3x3",
                "--cls-loss", "qfl",
                "--scale-weights", "3.0,2.0,1.0,0.5",
                "--weights", str(args.weights),
                "--data", str(Path(args.data) / "data.yaml"),
                "--device", str(args.device),
                "--output", str(OUTPUTS_DIR / "phase5_best_p2_combination"),
                "--topk", str(best_inf_topk),
                "--p2-conf", str(best_inf_conf),
                "--fusion-iou", str(best_inf_iou),
            ]
            if args.smoke:
                cmd_p5.append("--smoke")
            elif args.allow_full:
                cmd_p5.append("--allow-full")
            if args.hf_upload:
                cmd_p5.append("--hf-upload")

            ret_p5 = run_cmd_logged(cmd_p5, "Phase 5: Best P2 Combination Training")
            if ret_p5 == 0:
                completed.append("Phase 5: Best P2 Combination")
                p5_best = OUTPUTS_DIR / "phase5_best_p2_combination" / "weights" / "best_p2.pt"
                if p5_best.is_file():
                    current_p2_ckpt = str(p5_best)
            else:
                log(f"[Phase 5 Error] Exited with code {ret_p5}")

        # ===================================================================
        # PHASE 6: Canonical Comprehensive Evaluation & Decision Gate
        # ===================================================================
        if 6 in selected_phases:
            update_progress("Phase 6", "Canonical Final Evaluation & Decision Gate", completed)
            cmd_p6 = [
                sys.executable,
                str(BASE_DIR / "eval_comparison.py"),
                "--checkpoint", str(current_p2_ckpt),
                "--weights", str(args.weights),
                "--data", str(args.data),
                "--device", str(args.device),
                "--output", str(OUTPUTS_DIR / "phase6_final_evaluation"),
                "--topk", str(best_inf_topk),
                "--p2-conf", str(best_inf_conf),
                "--fusion-iou", str(best_inf_iou),
            ]
            if args.smoke:
                cmd_p6.extend(["--num-images", "10"])
            if args.hf_upload:
                cmd_p6.append("--hf-upload")

            ret_p6 = run_cmd_logged(cmd_p6, "Phase 6: Final Evaluation & Decision Gate")
            if ret_p6 == 0:
                completed.append("Phase 6: Final Evaluation & Decision Gate")
            else:
                log(f"[Phase 6 Error] Exited with code {ret_p6}")

        update_progress("Complete", "All selected phases finished", completed, status="completed")
        log("\n" + "=" * 80)
        log("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        log(f"Completed Tasks: {completed}")
        log(f"All outputs stored in: {OUTPUTS_DIR}")
        log("=" * 80)

    except Exception as exc:
        log(f"[FATAL PIPELINE ERROR] {exc}")
        update_progress("Failed", str(exc), completed, status="failed")
        return 1
    finally:
        stop_heartbeat.set()

    return 0


if __name__ == "__main__":
    sys.exit(main())
