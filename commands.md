# HRP4K CLI Commands Reference

## Primary Commands

### `hrp4k setup`
One-command environment setup: dependencies, dataset, HF credentials.

```bash
hrp4k setup
hrp4k setup --skip-dataset  # Skip dataset download
```

### `hrp4k experiment <name>`
Run any official experiment by name. Auto-resumes from HF if previous state exists.

```bash
# List all experiments
hrp4k experiment list

# Resolution experiments
hrp4k experiment yolo11m-resolution-4k
hrp4k experiment yolo11m-resolution-2k
hrp4k experiment yolo11m-resolution-1k
hrp4k experiment yolo11m-resolution-640
hrp4k experiment rtdetr-l-resolution-4k
hrp4k experiment rtdetr-l-resolution-2k
hrp4k experiment rtdetr-l-resolution-1k
hrp4k experiment rtdetr-l-resolution-640

# Slicing experiments (inference-only, frozen 640 checkpoint)
hrp4k experiment yolo11m-slicing-full
hrp4k experiment yolo11m-slicing-sliced-nms
hrp4k experiment yolo11m-slicing-sahi
hrp4k experiment yolo11m-slicing-perspective-grid
hrp4k experiment rtdetr-l-slicing-full
hrp4k experiment rtdetr-l-slicing-sliced-nms
hrp4k experiment rtdetr-l-slicing-sahi
hrp4k experiment rtdetr-l-slicing-perspective-grid

# Dry run (show config without executing)
hrp4k experiment yolo11m-resolution-640 --dry-run

# Custom frozen checkpoint for slicing
hrp4k experiment yolo11m-slicing-sahi --frozen-checkpoint path/to/best.pt
```

## Secondary Commands

### `hrp4k train`
Direct baseline training (prefer `experiment` command).

```bash
hrp4k train --model yolo11m --imgsz original --batch 16 --epochs 150 --allow-full
hrp4k train --model rtdetr-l --imgsz 640 --batch 16 --smoke
```

### `hrp4k predict`
Direct inference with spatial methods.

```bash
hrp4k predict --detector yolo11m --weights best.pt --method sliced-nms --split test
```

### `hrp4k evaluate`
Standalone COCO evaluation.

```bash
hrp4k evaluate --ground-truth HRP4K/test.json --predictions predictions.json --output metrics.json
```

### `hrp4k push-hf`
Upload local artifacts to Hugging Face.

```bash
hrp4k push-hf --token $HF_TOKEN --path outputs/
```

### `hrp4k status`
Show registered experiments and detector status.

### `hrp4k preflight`
Verify dataset identity, integrity, and runtime dependencies.

### `hrp4k prepare-dataset`
Create full dataset view with symlinks.

### `hrp4k prepare-smoke`
Create minimal smoke dataset view.

### `hrp4k run-smoke`
Run end-to-end smoke pipeline.

```bash
hrp4k run-smoke --data HRP4K --device cpu
```
