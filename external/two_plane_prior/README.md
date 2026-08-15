# Two-Plane Prior (TPP) External Adaptation

Learned Two-Plane Perspective Prior based Image Resampling for Efficient Object Detection on HRP4K.

## Upstream Reference
* **Repository:** `geometriczoom/two-plane-prior`
* **Environment:** Python 3.8.5, PyTorch 1.6.0, MMCV 1.3.17, MMDetection 2.20.0, Kornia 0.5.11

## Perspective Adaptation Protocol
1. Estimate ground-plane / vanishing-point parameters from training set geometry or road vanishing point estimator (no test annotation leakage).
2. Generate learnable warp grid allocating higher spatial sampling density to the far road plane.
3. Perform detector inference on warped canvas.
4. Invert bounding box coordinates back to canonical 4K original pixel coordinates using inverse warp map.
5. Export canonical HRP4K JSON predictions via `external/common/canonical.py`.
