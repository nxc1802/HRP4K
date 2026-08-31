# HRP4K — High-Resolution Pothole 4K Benchmark

Reproducible research benchmark for pothole detection on 4K UHD road imagery.

## Quick Start

```bash
# 1. One-command setup
hrp4k setup

# 2. Run any experiment
hrp4k experiment yolo11m-resolution-640
hrp4k experiment rtdetr-l-resolution-4k
hrp4k experiment yolo11m-slicing-sliced-nms
```

## Research Scope

Two detectors only:
- **YOLO11m** — Ultralytics YOLOv11 Medium (CNN)
- **RT-DETR-L** — Ultralytics Real-Time DEtection TRansformer Large (32.8M params)

Three experiment phases:

### Phase 1 — Resolution
Train both detectors at 4K, 2K, 1K, and 640 resolution.

### Phase 2 — Spatial Decomposition / Slicing
Inference-only experiments using frozen 640 checkpoint:
- Full Image (Baseline)
- Sliced-NMS
- SAHI
- Perspective Grid

### Phase 3 — Proposed Method
Pipeline skeleton only (no training, no benchmark, no claimed results).

## CLI Commands

| Command | Description |
| :--- | :--- |
| `hrp4k setup` | One-command environment setup |
| `hrp4k experiment <name>` | Run an official experiment |
| `hrp4k experiment list` | List all registered experiments |
| `hrp4k status` | Show detector and experiment status |
| `hrp4k preflight` | Verify dataset and runtime |
| `hrp4k train` | Direct training (prefer `experiment`) |
| `hrp4k predict` | Direct inference (prefer `experiment`) |
| `hrp4k evaluate` | Standalone COCO evaluation |
| `hrp4k push-hf` | Upload artifacts to Hugging Face |

## Experiment Matrix

```
hrp4k experiment list
```

### Resolution (8 experiments)
```
yolo11m-resolution-4k      rtdetr-l-resolution-4k
yolo11m-resolution-2k      rtdetr-l-resolution-2k
yolo11m-resolution-1k      rtdetr-l-resolution-1k
yolo11m-resolution-640     rtdetr-l-resolution-640
```

### Slicing (8 experiments)
```
yolo11m-slicing-full              rtdetr-l-slicing-full
yolo11m-slicing-sliced-nms        rtdetr-l-slicing-sliced-nms
yolo11m-slicing-sahi              rtdetr-l-slicing-sahi
yolo11m-slicing-perspective-grid  rtdetr-l-slicing-perspective-grid
```

## Dataset

**HRP4K** — 6,003 images (3840×2160, 16:9)
- Train: 4,202 images
- Valid: 901 images
- Test: 900 images (600 positive + 300 negative)

## Hugging Face Storage

All experiment artifacts are automatically synced to:
[Cuong2004/HRP4K](https://huggingface.co/datasets/Cuong2004/HRP4K)

## Installation

```bash
pip install -e ".[vision]"
```

## License

MIT
