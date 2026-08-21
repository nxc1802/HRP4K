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
| ZoomDet | learned non-uniform 2D grid warp | yes/no | ready |

`perspective-grid` uses hand-designed 3-band vertical geometry with 2D overlap, `sliced-nms` uses uniform sliding window, and `zoomdet` uses continuous non-uniform 2D grid deformation (`GridWarpTransform`).

Native crop methods use `IdentityTransform` or `CropTransform`. FOVEA/TPP adapters share `SeparableWarpTransform`; ZoomDet adapters use `GridWarpTransform`. Boxes are mapped by corners and round-trip transforms pass unit tests with near-zero error.
