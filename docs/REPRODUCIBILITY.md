# Reproducibility protocol

## Dataset identity

The downloaded HRP4K release is the project's single official dataset version. Its `train.json`, `valid.json` and `test.json` SHA-256 values are frozen in `dataset_identity.py`. Missing train image files originate from the release source; the project intends to request a complete archive from the authors. `prepare-dataset` without limits creates the official available release view. A limited `prepare-smoke` view remains labelled `smoke` even when its source hashes match.

Run `hrp4k preflight --data HRP4K --require-official` before an official experiment.

## Training

Full training requires `--allow-full`, a verified official manifest and the unbounded official dataset view. Smoke training is limited to at most two epochs and is never a scientific result. The resolved config, dataset manifest, package environment, seed and Git commit are retained.

## Evaluation

The evaluator accepts only canonical COCO predictions. It rejects unknown images/categories, malformed or non-finite boxes, non-positive dimensions, and invalid scores. No record is clamped or silently discarded. Official COCO AP uses pycocotools when installed; scale metrics and negative-image FPPI are additionally reported.

## Latency

CUDA is synchronized before and after timed regions. Default benchmark warm-up is 20 images; smoke uses one image. End-to-end latency includes decode, method preparation, detector calls, coordinate remapping and fusion. Reports store mean/median/P95/std end-to-end latency, processor/detector means, peak VRAM, device, GPU, PyTorch/CUDA/framework versions, precision, batch size, image size, warm-up and Git commit.

Latency values are comparable only when hardware, precision, batch size and framework protocol match.

## Leakage boundary

Dataset priors and calibration for learned methods may use train annotations only. Test annotations are used exclusively by evaluation. Phase 2 comparisons keep the detector architecture and checkpoint protocol fixed.
