# External reproduction contracts

External detector and learned-method repositories stay outside the core Python environment to avoid dependency conflicts. Every runner must export canonical COCO detections with `image_id`, `category_id`, `bbox` in source-image XYWH coordinates, and `score` in `[0,1]`. Validate exports with `hrp4k evaluate` before diagnostics.

An external method is never marked ready merely because this contract exists. Its directory records the required upstream implementation and expected hand-off.
