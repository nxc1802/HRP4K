#!/usr/bin/env python3
"""CLI Entry Point for Phase 1: Zero-Compute Inference-Time Optimization Sweep.

Sweeps Top-K, P2 Confidence Threshold, and Fusion NMS IoU without re-training:
  - Top-K: [300, 500, 1000, 2000]
  - P2 Threshold: [0.001, 0.003, 0.005, 0.01, 0.02]
  - NMS IoU: [0.4, 0.5, 0.6, 0.7]

Usage:
  python3 sweep_p2_inference.py --checkpoint outputs/experiments/rtdetr-l-proposed-p2-2k/weights/best_p2.pt --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hrp4k.experiments.sweep_inference import run_inference_sweep


def main() -> int:
    parser = argparse.ArgumentParser(description="HRP4K Phase 1: Inference Optimization Sweep")
    parser.add_argument(
        "--checkpoint",
        default="outputs/experiments/rtdetr-l-proposed-p2-2k/weights/best_p2.pt",
        help="Path to trained P2 head checkpoint (e.g. best_p2.pt)",
    )
    parser.add_argument(
        "--weights",
        default="outputs/experiments/rtdetr-l-resolution-2k/weights/best.pt",
        help="Path to base fine-tuned RT-DETR model checkpoint",
    )
    parser.add_argument(
        "--data",
        default="HRP4K",
        help="Path to HRP4K dataset directory containing test.json",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1920,
        help="Canonical image size (default: 1920)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device index (e.g. '0') or 'cpu'",
    )
    parser.add_argument(
        "--output",
        default="outputs/inference_sweep",
        help="Output directory for sweep results and best config",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=0,
        help="Number of images to evaluate (0 = all images in test set)",
    )
    parser.add_argument(
        "--rect",
        action="store_true",
        help="Use rectangular letterbox instead of canonical square 1920x1920",
    )
    parser.add_argument(
        "--topk",
        type=str,
        default="300,500,1000,2000",
        help="Comma-separated Top-K values to sweep",
    )
    parser.add_argument(
        "--conf",
        type=str,
        default="0.001,0.003,0.005,0.01,0.02",
        help="Comma-separated P2 confidence thresholds to sweep",
    )
    parser.add_argument(
        "--iou",
        type=str,
        default="0.4,0.5,0.6,0.7",
        help="Comma-separated NMS IoU thresholds to sweep",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for parallel GPU inference (default: 8, optimal for 80/96GB GPU)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for parallel image prefetching (default: 4)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable FP16 Tensor Core acceleration and force FP32",
    )
    parser.set_defaults(fp16=True)
    parser.add_argument(
        "--hf-upload",
        action="store_true",
        help="Upload sweep results to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-repo",
        default="Cuong2004/HRP4K",
        help="Hugging Face repo ID",
    )

    args = parser.parse_args()

    topk_list = [int(x.strip()) for x in args.topk.split(",") if x.strip()]
    conf_list = [float(x.strip()) for x in args.conf.split(",") if x.strip()]
    iou_list = [float(x.strip()) for x in args.iou.split(",") if x.strip()]

    res = run_inference_sweep(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        weights=args.weights,
        image_size=args.imgsz,
        device=args.device,
        output_dir=args.output,
        num_images=args.num_images,
        rect=args.rect,
        topk_candidates=topk_list,
        conf_candidates=conf_list,
        iou_candidates=iou_list,
        batch_size=args.batch_size,
        num_workers=args.workers,
        fp16=args.fp16,
    )

    if args.hf_upload:
        try:
            from hrp4k.infra.upload import get_hf_credentials
            from huggingface_hub import HfApi
            token, repo_id, repo_type = get_hf_credentials(repo_id=args.hf_repo)
            if token and repo_id:
                api = HfApi(token=token)
                api.upload_folder(
                    folder_path=str(args.output),
                    path_in_repo=str(args.output),
                    repo_id=repo_id,
                    repo_type=repo_type,
                    commit_message=f"Sync Phase 1 sweep results from {args.output}",
                )
                print(f"[HF Sync] Successfully uploaded {args.output} to {repo_id} on Hugging Face Hub")
            else:
                print("[HF Sync] Warning: HF credentials not configured, skipping upload.")
        except Exception as exc:
            print(f"[HF Sync] Warning: Could not upload to HF: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
