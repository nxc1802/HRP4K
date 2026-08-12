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
| Two-Plane Prior | learned perspective warp | yes | external required |
| ZoomDet | learned non-uniform grid warp | yes | external required |

`perspective-grid` is not TPP, and `sliced-nms` is not SAHI. Learned methods remain unavailable until their official/paper-faithful runtime and checkpoints are connected; the CLI fails instead of substituting heuristics.

Native crop methods use `IdentityTransform` or `CropTransform`. FOVEA/TPP adapters share `SeparableWarpTransform`; ZoomDet-style adapters use `GridWarpTransform`. Boxes are mapped by corners and round-trip transforms must pass tests.
