# Changelog

## 0.4.0

- Recognize the hash-verified downloaded HRP4K release as the single official dataset version while keeping bounded smoke views explicitly non-benchmark.
- Add fail-fast canonical prediction validation shared by evaluation and diagnostics.
- Add CUDA-synchronized timing, configurable warm-up, runtime provenance and richer latency statistics.
- Route Ultralytics through a framework-agnostic detector runner with versioned experiment manifests and deterministic experiment IDs.
- Add identity/crop/separable/grid coordinate transforms with round-trip tests.
- Add structured method registry, optional official SAHI integration, config-driven `run`, and `preflight` CLI commands.
- Document external contracts for RT-DETR, D-FINE and learned Phase 2 reproductions without claiming unexecuted methods.

## 0.3.0

- Validate diagnostic input schemas and ignore metrics/per-image JSON accidentally supplied by broad wildcards.
- Require complete train and validation splits for official training, and train/validation/test for an official benchmark.
- Make `prepare-dataset` select all locally available images by default; retain bounded `prepare-smoke` behavior.
- Split inference timing into decode, processor, detector, fusion and end-to-end latency.
- Rename detector-pixel estimates to nominal detector canvas budget and report nominal compute amplification.
- Add a lightweight pycocotools integration CI job.
- Add regression coverage for completeness, all-available preparation and wildcard filtering.

## 0.2.0

- Add explicit SGD training and full/incomplete-training safeguards.
- Correct official negative-set FPPI and infer COCO category IDs from ground truth.
- Add `DetectorAdapter`, `perspective-grid`, nominal compute metrics and Phase 3 paired/Pareto diagnostics.
- Add Phase 0 split-shift statistics, dataset manifests/hashes, dependency lock, tests and CI.
