# Changelog

## 0.5.0 (Upgrade 3.0)

- Modernize repository to `src/` layout with `src/hrp4k/` and clean backward-compatibility shim `src/hrp4k_suite/`.
- Introduce 5 core first-class abstractions: Config, Experiment, Artifact, Phase, and Scientific Contract.
- Implement modular YAML composition subsystem (`base.yaml` + `detectors/` + `methods/` + `profiles/` + CLI overrides → `resolved_config.yaml`).
- Add `hrp4k config show`, `hrp4k config validate`, and `hrp4k experiment id` CLI commands.
- Decompose monolithic `dataset.py`, `processing.py`, and `runner.py` into dedicated sub-packages (`data/`, `detectors/`, `methods/`, `inference/`, `evaluation/`, `phases/`, `diagnostics/`, `protocol/`, `infra/`).
- Implement 4-tier test architecture (`tests/unit/`, `tests/contracts/`, `tests/scientific/`, `tests/integration/`) with 37 automated tests passing.
- Reorganize `docs/` into 5 structured categories (`paper/`, `architecture/`, `phases/`, `methodology/`, `guides/`) with a Master Documentation Hub.
- Update `pyproject.toml` with `src` package discovery, dev dependencies, and pytest configuration.

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
