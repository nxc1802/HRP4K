import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from torch.utils.data import DataLoader

from hrp4k.models.scout import (
    MobileNetV3Scout,
    CandidateGenerator,
    evaluate_scout_regions,
)
from hrp4k.training.scout_trainer import ScoutDataset, _collate_fn


def main():
    weights_p = "/marimo/HRP4K/outputs/runs/scout/weights/scout_best.pt"
    data_p = "/marimo/HRP4K/HRP4K"
    out_p = "/marimo/HRP4K/outputs/runs/scout/sweep_results.json"

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Sweep Optimizer] Loading model onto {dev}...")

    model = MobileNetV3Scout().to(dev)
    ckpt = torch.load(weights_p, map_location=dev, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    th_list = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    k_list = [1, 2, 3, 4, 6]

    results = {
        "valid": {"threshold_sweep": {}, "k_sweep": {}},
        "test": {"threshold_sweep": {}, "k_sweep": {}},
    }

    for split in ["valid", "test"]:
        print(f"\n================== PRE-COMPUTING HEATMAPS: {split.upper()} (900 images) ==================")
        ds = ScoutDataset(data_p, split=split, augment=False)
        loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=_collate_fn, num_workers=0)

        all_heatmaps = []
        all_orig_boxes = []
        all_orig_sizes = []

        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(dev)
                with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                    preds = model(imgs)
                pred_np = preds.float().cpu().numpy()

                for i in range(len(batch["image_ids"])):
                    all_heatmaps.append(pred_np[i, 0])
                    all_orig_boxes.append(batch["orig_boxes"][i])
                    all_orig_sizes.append(batch["orig_sizes"][i])

        dt_infer = time.time() - t0
        print(f"Pre-computed {len(all_heatmaps)} heatmaps in {dt_infer:.2f}s ({len(all_heatmaps)/dt_infer:.1f} img/s)!")

        # 1. Threshold sweep at K=4
        print("\n--- Threshold Sweep (K=4, Margin=30%) ---")
        for th in th_list:
            gen = CandidateGenerator(threshold=th, context_margin=0.30, k_max=4)
            recalls, coverages, false_rates, k_crops = [], [], [], []
            t_s = time.time()
            for i in range(len(all_heatmaps)):
                hmap = all_heatmaps[i]
                orig_w, orig_h = all_orig_sizes[i]
                boxes = all_orig_boxes[i]

                cands = gen.generate(hmap, source_width=orig_w, source_height=orig_h)
                res = evaluate_scout_regions(boxes, cands)
                recalls.append(res["region_recall"])
                coverages.append(res["gt_coverage_ratio"])
                false_rates.append(res["false_region_rate"])
                k_crops.append(res["k_crops"])

            mean_rec = float(np.mean(recalls))
            mean_cov = float(np.mean(coverages))
            mean_false = float(np.mean(false_rates))
            mean_k = float(np.mean(k_crops))

            key = f"tau_{th:.2f}"
            results[split]["threshold_sweep"][key] = {
                "region_recall": mean_rec,
                "gt_coverage": mean_cov,
                "false_region_rate": mean_false,
                "avg_k": mean_k,
            }
            print(f"{split} | tau={th:.2f}: Recall={mean_rec*100:.2f}%, GT_Cov={mean_cov*100:.2f}%, False={mean_false*100:.1f}%, Avg_K={mean_k:.2f} ({time.time()-t_s:.2f}s)")

        # 2. K sweep at tau=0.05
        print("\n--- K-Budget Sweep (tau=0.05, Margin=30%) ---")
        for k in k_list:
            gen = CandidateGenerator(threshold=0.05, context_margin=0.30, k_max=k)
            recalls, coverages, false_rates, k_crops = [], [], [], []
            t_s = time.time()
            for i in range(len(all_heatmaps)):
                hmap = all_heatmaps[i]
                orig_w, orig_h = all_orig_sizes[i]
                boxes = all_orig_boxes[i]

                cands = gen.generate(hmap, source_width=orig_w, source_height=orig_h)
                res = evaluate_scout_regions(boxes, cands)
                recalls.append(res["region_recall"])
                coverages.append(res["gt_coverage_ratio"])
                false_rates.append(res["false_region_rate"])
                k_crops.append(res["k_crops"])

            mean_rec = float(np.mean(recalls))
            mean_cov = float(np.mean(coverages))
            mean_false = float(np.mean(false_rates))
            mean_k = float(np.mean(k_crops))

            key = f"k_{k}"
            results[split]["k_sweep"][key] = {
                "region_recall": mean_rec,
                "gt_coverage": mean_cov,
                "false_region_rate": mean_false,
                "avg_k": mean_k,
            }
            print(f"{split} | K_max={k}: Recall={mean_rec*100:.2f}%, GT_Cov={mean_cov*100:.2f}%, False={mean_false*100:.1f}%, Avg_K={mean_k:.2f} ({time.time()-t_s:.2f}s)")

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Completed All Sweeps! Results saved to {out_p}")


if __name__ == "__main__":
    main()
