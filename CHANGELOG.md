# Changelog

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
