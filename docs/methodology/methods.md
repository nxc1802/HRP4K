# Frozen benchmark methods

All methods emit canonical COCO detections in original-image coordinates. Phase 1 compares detectors with `resize`; Phase 2 holds the detector protocol constant while changing resolution allocation.

| Method | Type | Training | Status |
|---|---|---:|---|
| Resize | single canvas | no | ready |
| Uniform-2 / Uniform-3 | crop grid + global NMS | no | ready |
| sliced-nms | in-house overlapping crops + global NMS | no | ready |
| SAHI | official optional library, NMS/IoU logged | no | optional-ready |
| perspective-grid | hand-designed geometry crops | no | ready |
| AutoFocus | learned coarse-to-fine focus chips | yes | external required |
| AdaZoom | learned adaptive zoom policy | yes | external required |
| FOVEA | nonlinear separable warp | yes | external required |
| ZoomDet (Neural) | official lightweight sub-network 2D grid warp | yes | ready |
| ZoomDet (Geometry) | dataset-prior road geometry continuous 2D warp | no/yes | ready |

`perspective-grid` uses hand-designed 3-band vertical geometry with 2D overlap, `sliced-nms` uses uniform sliding window, and `zoomdet` provides two continuous 2D deformation options:
1. **Option 1 (`zoomdet-neural` - Official)**: Lightweight `NeuralZoomGenerator` ConvNet predicting dynamic 2D displacement offsets.
2. **Option 2 (`zoomdet-geometry` - In-house Road Prior)**: Parametric non-linear 2D deformation grid derived from HRP4K spatial dataset analysis (compressing sky $0-40\%$ and expanding far road $40-75\%$).

Native crop methods use `IdentityTransform` or `CropTransform`. FOVEA/TPP adapters share `SeparableWarpTransform`; ZoomDet adapters use `GridWarpTransform`. Boxes are mapped by corners and round-trip transforms pass unit tests with near-zero error.
