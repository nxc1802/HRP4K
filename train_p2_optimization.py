#!/usr/bin/env python3
"""Dedicated Training Entry Point for P2 Head Optimization Phases (Phases 2 - 5).

Supports:
  - Phase 2: Multi-Positive Target Assignment (1x1 vs 3x3)
  - Phase 3: Classification Loss Function (BCE vs Focal Loss vs QFL)
  - Phase 4: Scale-Aware Loss Weighting (Ultra-fine / Fine / Medium / Large)
  - Phase 5: Build Best P2 Combination (Best assignment + Best loss + Best scale weights)

Usage Examples:
  # Phase 2: Multi-positive 3x3
  python3 train_p2_optimization.py --phase 2 --target-assignment 3x3 --allow-full --device 0

  # Phase 3: Quality Focal Loss
  python3 train_p2_optimization.py --phase 3 --target-assignment 3x3 --cls-loss qfl --allow-full --device 0

  # Phase 4: Scale-aware weighting
  python3 train_p2_optimization.py --phase 4 --target-assignment 3x3 --cls-loss qfl --scale-weights 3.0,2.0,1.0,0.5 --allow-full --device 0

  # Phase 5: Final best combination
  python3 train_p2_optimization.py --phase 5 --target-assignment 3x3 --cls-loss qfl --scale-weights 3.0,2.0,1.0,0.5 --topk 1000 --p2-conf 0.005 --allow-full --device 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add src/ to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hrp4k.experiments.proposed import train_rtdetr_p2


def main() -> int:
    parser = argparse.ArgumentParser(description="HRP4K P2 Optimization Training Runner (Phases 2-5)")
    parser.add_argument(
        "--phase",
        type=int,
        default=2,
        choices=[2, 3, 4, 5],
        help="Optimization phase: 2 (Multi-positive), 3 (Loss), 4 (Scale-aware), 5 (Best combination)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom experiment name suffix (default: auto-generated from phase and parameters)",
    )
    parser.add_argument(
        "--target-assignment",
        type=str,
        default="1x1",
        choices=["1x1", "3x3"],
        help="Target assignment strategy: '1x1' (single center cell) or '3x3' (center 3x3 region)",
    )
    parser.add_argument(
        "--cls-loss",
        type=str,
        default="bce",
        choices=["bce", "focal", "qfl"],
        help="Classification loss type: 'bce', 'focal' (gamma=2.0), or 'qfl' (Quality Focal Loss)",
    )
    parser.add_argument(
        "--scale-weights",
        type=str,
        default=None,
        help="Comma-separated 4 floats: Ultra-fine, Fine, Medium, Large (e.g. '3.0,2.0,1.0,0.5')",
    )
    parser.add_argument(
        "--weights",
        default="outputs/experiments/rtdetr-l-resolution-2k/weights/best.pt",
        help="Base fine-tuned RT-DETR-L checkpoint",
    )
    parser.add_argument(
        "--data",
        default="HRP4K/data.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs (default: 150)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1920,
        help="Image size (default: 1920)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device index or 'cpu'",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=300,
        help="Inference Top-K predictions",
    )
    parser.add_argument(
        "--p2-conf",
        type=float,
        default=0.001,
        help="P2 confidence score threshold for inference",
    )
    parser.add_argument(
        "--fusion-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold for prediction fusion",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: outputs/experiments/p2_phase_<phase>_<name>)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 epoch smoke verification",
    )
    parser.add_argument(
        "--allow-full",
        action="store_true",
        help="Explicit flag required for full training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoint",
    )
    parser.add_argument(
        "--p2-checkpoint",
        default=None,
        help="Optional existing P2 checkpoint to resume or warm-start from",
    )
    parser.add_argument(
        "--rect",
        action="store_true",
        help="Use rectangular letterbox instead of square",
    )
    parser.add_argument(
        "--hf-upload",
        action="store_true",
        help="Upload checkpoints and metrics to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-repo",
        default=os.environ.get("HF_REPO", "Cuong2004/HRP4K"),
        help="Hugging Face repo id",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face write token",
    )

    args = parser.parse_args()

    # Parse scale weights if provided
    scale_w_tuple = None
    if args.scale_weights:
        parts = [float(x.strip()) for x in args.scale_weights.split(",") if x.strip()]
        if len(parts) == 4:
            scale_w_tuple = tuple(parts)
        else:
            raise ValueError(f"--scale-weights must have 4 values (Ultra-fine, Fine, Medium, Large), got {args.scale_weights}")

    # Build experiment folder name
    suffix = args.name or f"assign_{args.target_assignment}_loss_{args.cls_loss}"
    if scale_w_tuple:
        suffix += f"_scale_{'_'.join(str(w) for w in scale_w_tuple)}"

    out_dir = Path(args.output) if args.output else Path(f"outputs/experiments/p2_phase_{args.phase}_{suffix}")

    print("=" * 80)
    print(f"HRP4K OPTIMIZATION: PHASE {args.phase}")
    print("=" * 80)
    print(f"  Target Assignment:  {args.target_assignment}")
    print(f"  Class Loss Type:    {args.cls_loss}")
    print(f"  Scale Weights:      {scale_w_tuple or '(1.0, 1.0, 1.0, 1.0) [Uniform]'}")
    print(f"  Inference Top-K:    {args.topk}")
    print(f"  P2 Conf Cutoff:     {args.p2_conf}")
    print(f"  Fusion NMS IoU:     {args.fusion_iou}")
    print(f"  Output Dir:         {out_dir}")
    print(f"  Base Model:         {args.weights}")
    print(f"  Epochs:             {args.epochs} (Patience: {args.patience})")
    print("=" * 80)

    res = train_rtdetr_p2(
        dataset_yaml=Path(args.data),
        weights=args.weights,
        run_dir=out_dir,
        smoke=args.smoke,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch=args.batch,
        accumulation=args.accumulation,
        patience=args.patience,
        device=args.device,
        allow_full=args.allow_full,
        resume=args.resume,
        p2_checkpoint=args.p2_checkpoint,
        rect=args.rect,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        hf_sync=args.hf_upload,
        target_assignment=args.target_assignment,
        cls_loss_type=args.cls_loss,
        scale_weights=scale_w_tuple,
        topk=args.topk,
        p2_conf_threshold=args.p2_conf,
        fusion_iou_threshold=args.fusion_iou,
    )

    print(f"\n[Training Finished] Run outputs saved to: {res.get('run_dir')}")
    print(f"Best P2 Checkpoint: {res.get('best')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
