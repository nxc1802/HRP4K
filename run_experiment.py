#!/usr/bin/env python3
"""HRP4K Experiment Runner Script.

Runs official HRP4K CLI commands via OS subprocess with real-time unbuffered logging,
crash resilience, and easy parameter customization.
"""

from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# ==============================================================================
# CONFIGURATION — CHỈNH SỬA THÔNG SỐ TẠI ĐÂY
# ==============================================================================
# Chế độ chạy:
#   - "train"         : Chạy trọn gói (Train P2-only + Val + Test Eval + Push HF)
#   - "full_finetune" : Chạy Full Fine-tune (Native RT-DETR-L + P2 Head, epochs=30, patience=5)
#   - "eval"          : Chỉ đánh giá checkpoint đã train trên tập test
#   - "compare"       : Đánh giá và so sánh song song P2-Only vs Native vs Fusion
#   - "inspect"       : Chỉ đọc và in metadata của checkpoint
#   - "calibration"   : Chạy chẩn đoán score calibration [0.001, 0.01, 0.05, 0.10, 0.25]
MODE = "full_finetune"

# Tên thí nghiệm (dùng khi MODE = "train" hoặc "full_finetune"):
# Options: "rtdetr-l-proposed-full-2k", "rtdetr-l-proposed-p2-2k", "rtdetr-l-proposed-p2-640", "rtdetr-l-proposed-p2-4k"
EXPERIMENT_NAME = "rtdetr-l-proposed-full-2k"

# Model Base RT-DETR đã fine-tune trên HRP4K (nếu chưa có trên máy, hệ thống tự động tải từ Hugging Face):
BASE_WEIGHTS = "outputs/experiments/rtdetr-l-resolution-2k/weights/best.pt"

# Có tiếp tục train checkpoint cũ hay train mới từ đầu?
#   - False: Ghép P2 head vào base model, train từ Epoch 1
#   - True (Resume): Tiếp tục checkpoint hiện tại
RESUME_P2 = False
P2_CHECKPOINT = ""

# Hyperparameters (dùng khi MODE = "train" hoặc "full_finetune"):
BATCH_SIZE = 16       # Full batch 16 cho GPU 80GB-102GB
EPOCHS = 30           # 30 epochs cho full fine-tune
PATIENCE = 5          # Dừng sớm nếu 5 epoch không giảm loss
DEVICE = "0"          # CUDA device index ("0", "1", ...) hoặc "cpu"

# Đường dẫn checkpoint khi MODE = "eval", "compare", hoặc "inspect":
CHECKPOINT_PATH = "outputs/experiments/rtdetr-l-proposed-full-2k/weights/best_full.pt"

# Hugging Face đồng bộ kết quả (lấy từ biến môi trường hoặc file .env):
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_REPO", "Cuong2004/HRP4K")
ENABLE_HF_SYNC = bool(HF_TOKEN)

# File ghi log:
LOG_FILE = "train_full_finetune.log"
# ==============================================================================


def build_command() -> list[str]:
    """Construct CLI argument list based on active configuration."""
    python_bin = sys.executable

    if MODE in ("train", "full_finetune"):
        target_exp = "rtdetr-l-proposed-full-2k" if MODE == "full_finetune" else EXPERIMENT_NAME
        target_epochs = 30 if MODE == "full_finetune" else EPOCHS
        target_patience = 5 if MODE == "full_finetune" else PATIENCE

        cmd = [
            python_bin, "-m", "hrp4k.cli", "experiment", target_exp,
            "--batch", str(BATCH_SIZE),
            "--epochs", str(target_epochs),
            "--patience", str(target_patience),
            "--weights", BASE_WEIGHTS,
            "--device", str(DEVICE),
        ]
        if RESUME_P2:
            cmd.append("--resume")
            if P2_CHECKPOINT and Path(P2_CHECKPOINT).is_file():
                cmd.extend(["--p2-checkpoint", str(P2_CHECKPOINT)])
        if ENABLE_HF_SYNC and HF_TOKEN:
            cmd.extend(["--hf-token", HF_TOKEN, "--hf-repo", HF_REPO])
        else:
            cmd.append("--no-hf-sync")
        return cmd

    elif MODE == "eval":
        cmd = [
            python_bin, "-m", "hrp4k.cli", "eval-proposed",
            "--checkpoint", CHECKPOINT_PATH,
            "--device", str(DEVICE),
        ]
        if ENABLE_HF_SYNC and HF_TOKEN:
            cmd.extend(["--hf-token", HF_TOKEN, "--hf-repo", HF_REPO])
        return cmd

    elif MODE == "compare":
        cmd = [
            python_bin, "eval_comparison.py",
            "--checkpoint", CHECKPOINT_PATH,
            "--weights", BASE_WEIGHTS,
            "--device", str(DEVICE),
        ]
        if ENABLE_HF_SYNC and HF_TOKEN:
            cmd.extend(["--hf-upload", "--hf-token", HF_TOKEN, "--hf-repo", HF_REPO])
        return cmd

    elif MODE == "inspect":
        return [python_bin, "-m", "hrp4k.cli", "inspect-checkpoint", CHECKPOINT_PATH]

    elif MODE == "calibration":
        return [
            python_bin, "check_calibration.py",
            "--checkpoint", CHECKPOINT_PATH,
            "--device", str(DEVICE),
            "--num-images", "20",
        ]

    else:
        raise ValueError(f"Unknown MODE: {MODE}. Choose 'train', 'eval', 'compare', 'inspect', or 'calibration'.")


def main() -> int:
    workspace_dir = Path(__file__).resolve().parent
    os.chdir(workspace_dir)

    # Set up environment variables
    env = os.environ.copy()
    src_dir = str(workspace_dir / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}:{existing_pythonpath}" if existing_pythonpath else src_dir
    env["PYTHONUNBUFFERED"] = "1"
    if DEVICE != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(DEVICE)
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
        env["HF_REPO"] = HF_REPO

    cmd = build_command()
    log_path = workspace_dir / LOG_FILE

    print("=" * 70)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HRP4K Python CLI Runner")
    print(f"  Mode:        {MODE}")
    print(f"  Command:     {' '.join(cmd)}")
    print(f"  Workspace:   {workspace_dir}")
    print(f"  Log File:    {log_path}")
    print("=" * 70)

    # Execute process and stream to both stdout and persistent log file
    with open(log_path, "a", encoding="utf-8") as f_log:
        f_log.write(f"\n\n{'=' * 70}\n")
        f_log.write(f"Run started at {datetime.now().isoformat()}\n")
        f_log.write(f"Command: {' '.join(cmd)}\n")
        f_log.write(f"{'=' * 70}\n\n")
        f_log.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        try:
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                sys.stdout.write(line)
                sys.stdout.flush()
                f_log.write(line)
                f_log.flush()
        except KeyboardInterrupt:
            print("\n[Runner] Interrupted by user. Terminating child process...")
            process.terminate()
            process.wait()
            return 130

        process.wait()
        f_log.write(f"\n[Runner] Process finished with exit code {process.returncode}\n")
        f_log.flush()

    print(f"\n[Runner] Execution completed with exit code {process.returncode}")
    print(f"[Runner] Full log saved to: {log_path}")
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
