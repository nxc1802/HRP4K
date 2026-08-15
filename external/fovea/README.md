# FOVEA External Adaptation

Foveated Image Magnification for Autonomous Navigation (ICCV 2021) on HRP4K.

## Upstream Reference
* **Repository:** `tchittesh/fovea`
* **Environment:** Python 3.8.5, PyTorch 1.6.0, MMDetection 2.7.0

## FOVEA Pipeline
1. Estimate spatial saliency prior over the training set (no test annotation leakage).
2. Generate continuous separable 1-D axis warps (`SeparableWarpTransform`).
3. Warp 4K input images onto a small fixed canvas.
4. Execute detector on warped canvas.
5. Invert bounding box coordinates back to canonical 4K original pixel coordinates using `SeparableWarpTransform.inverse_boxes`.
6. Export canonical HRP4K JSON predictions via `external/common/canonical.py`.
