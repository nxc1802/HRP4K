import sys
import json
import time
from pathlib import Path
import torch

from hrp4k.training.scout_trainer import evaluate_scout_model


def main():
    weights_p = "/marimo/HRP4K/outputs/runs/scout/weights/scout_best.pt"
    data_p = "/marimo/HRP4K/HRP4K"
    out_p = "/marimo/HRP4K/outputs/runs/scout/sweep_results.json"

    th_list = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    k_list = [1, 2, 3, 4, 6]

    results = {
        "valid": {"threshold_sweep": {}, "k_sweep": {}},
        "test": {"threshold_sweep": {}, "k_sweep": {}},
    }

    for split in ["valid", "test"]:
        print(f"\n================== SPLIT: {split.upper()} (900 images) ==================")
        # 1. Threshold sweep at K=4
        print("--- Threshold Sweep (K=4, Margin=30%) ---")
        for th in th_list:
            t0 = time.time()
            res = evaluate_scout_model(
                data_dir=data_p,
                weights_path=weights_p,
                split=split,
                threshold=th,
                context_margin=0.30,
                k_max=4,
                device="cuda:0",
            )
            key = f"tau_{th:.2f}"
            results[split]["threshold_sweep"][key] = {
                "region_recall": res["mean_region_recall"],
                "gt_coverage": res["mean_gt_coverage"],
                "false_region_rate": res["mean_false_region_rate"],
                "avg_k": res["mean_k"],
            }
            dt = time.time() - t0
            recall_pct = res["mean_region_recall"] * 100
            cov_pct = res["mean_gt_coverage"] * 100
            false_pct = res["mean_false_region_rate"] * 100
            avg_k = res["mean_k"]
            print(f"{split} | tau={th:.2f}: Recall={recall_pct:.2f}%, GT_Cov={cov_pct:.2f}%, False={false_pct:.1f}%, Avg_K={avg_k:.2f} ({dt:.1f}s)")

        # 2. K sweep at tau=0.05
        print("\n--- K-Budget Sweep (tau=0.05, Margin=30%) ---")
        for k in k_list:
            t0 = time.time()
            res = evaluate_scout_model(
                data_dir=data_p,
                weights_path=weights_p,
                split=split,
                threshold=0.05,
                context_margin=0.30,
                k_max=k,
                device="cuda:0",
            )
            key = f"k_{k}"
            results[split]["k_sweep"][key] = {
                "region_recall": res["mean_region_recall"],
                "gt_coverage": res["mean_gt_coverage"],
                "false_region_rate": res["mean_false_region_rate"],
                "avg_k": res["mean_k"],
            }
            dt = time.time() - t0
            recall_pct = res["mean_region_recall"] * 100
            cov_pct = res["mean_gt_coverage"] * 100
            false_pct = res["mean_false_region_rate"] * 100
            avg_k = res["mean_k"]
            print(f"{split} | K_max={k}: Recall={recall_pct:.2f}%, GT_Cov={cov_pct:.2f}%, False={false_pct:.1f}%, Avg_K={avg_k:.2f} ({dt:.1f}s)")

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Completed All Sweeps! Results saved to {out_p}")


if __name__ == "__main__":
    main()
