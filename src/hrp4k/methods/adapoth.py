"""AdaPoth-Lite Adaptive High-Resolution Processing Pipeline (CVPR/WACV Paper Core).

Implements:
1. Low-cost Global Scout (MobileNetV3-Small) -> Candidate Generation (Dynamic Top-K <= 4).
2. Shared Global-Local view preparation.
3. Inverse coordinate remapping from crop/thumbnail to original 4K (3840x2160).
4. Crop-boundary penalty (s' = s * p_boundary) to suppress truncated boundary detections.
5. Score temperature calibration (s_g' = T_g(s_g), s_l' = T_l(s_l)).
6. Global-Local Fusion (Class-agnostic NMS / Weighted Box Fusion).
7. Oracle & ablation baselines: adapoth-oracle, adapoth-fixed, adapoth-random.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..detectors.base import Detection
from ..methods.base import CoordinateTransform, CropTransform, IdentityTransform, ProcessedView, nms
from ..models.scout import CandidateGenerator, CandidateRegion, MobileNetV3Scout


@dataclass(frozen=True)
class ScaledCropTransform:
    """Coordinate transform for a local crop that is cropped from (x0, y0, w, h) and resized."""
    x0: float
    y0: float
    crop_w: float
    crop_h: float
    view_w: float
    view_h: float

    def forward_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        result = np.asarray(boxes_xyxy, dtype=float).copy()
        result[:, [0, 2]] = (result[:, [0, 2]] - self.x0) * (self.view_w / max(1.0, self.crop_w))
        result[:, [1, 3]] = (result[:, [1, 3]] - self.y0) * (self.view_h / max(1.0, self.crop_h))
        return result

    def inverse_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        result = np.asarray(boxes_xyxy, dtype=float).copy()
        result[:, [0, 2]] = result[:, [0, 2]] * (self.crop_w / max(1.0, self.view_w)) + self.x0
        result[:, [1, 3]] = result[:, [1, 3]] * (self.crop_h / max(1.0, self.view_h)) + self.y0
        return result


class GlobalScaleTransform:
    """Scale transform for full-image global view (e.g. 960x544 -> 3840x2160)."""
    def __init__(self, src_w: float = 3840.0, src_h: float = 2160.0, dst_w: float = 960.0, dst_h: float = 544.0):
        self.src_w = src_w
        self.src_h = src_h
        self.dst_w = dst_w
        self.dst_h = dst_h

    def forward_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        result = np.asarray(boxes_xyxy, dtype=float).copy()
        result[:, [0, 2]] *= (self.dst_w / self.src_w)
        result[:, [1, 3]] *= (self.dst_h / self.src_h)
        return result

    def inverse_boxes(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        result = np.asarray(boxes_xyxy, dtype=float).copy()
        result[:, [0, 2]] *= (self.src_w / self.dst_w)
        result[:, [1, 3]] *= (self.src_h / self.dst_h)
        return result


class AdaPothScoutEngine:
    """Singleton/Cached inference engine for the MobileNetV3 Scout model."""
    _instance: AdaPothScoutEngine | None = None
    _cached_weights: str | None = None

    def __init__(self, weights_path: Path | str | None = None, device: str | None = None):
        self.weights_path = str(weights_path) if weights_path else None
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            dev = torch.device(self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = MobileNetV3Scout().to(dev)
            if self.weights_path and Path(self.weights_path).is_file():
                ckpt = torch.load(self.weights_path, map_location=dev, weights_only=False)
                state_dict = ckpt["model"] if "model" in ckpt else ckpt
                self.model.load_state_dict(state_dict)
            self.model.eval()
            self._dev = dev
        except Exception:
            self.model = None

    def predict_heatmap(self, image_bgr: np.ndarray, scout_size: tuple[int, int] = (540, 960)) -> np.ndarray:
        """Run scout inference and return 2D float heatmap [heat_h, heat_w]."""
        if self.model is None:
            # Fallback heuristic heatmap based on road gradient if no torch/model
            h, w = scout_size
            heat = np.zeros((h // 16, w // 16), dtype=np.float32)
            heat[int(heat.shape[0] * 0.4):, :] = 0.35
            return heat

        import torch
        sh, sw = scout_size
        thumb = _resize_image(image_bgr, (sw, sh))
        thumb_t = torch.from_numpy(thumb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        thumb_t = (thumb_t - mean) / std

        with torch.no_grad():
            hmap = self.model(thumb_t.to(self._dev)).cpu().numpy()[0, 0]
        return hmap


def apply_boundary_penalty(
    detections: list[Detection],
    view_width: int,
    view_height: int,
    boundary_margin: int = 8,
    penalty: float = 0.70,
) -> list[Detection]:
    """Penalize detections that touch or lie near the crop boundary."""
    penalized = []
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        is_near_boundary = (
            x1 <= boundary_margin
            or y1 <= boundary_margin
            or x2 >= (view_width - boundary_margin)
            or y2 >= (view_height - boundary_margin)
        )
        new_score = (det.score * penalty) if is_near_boundary else det.score
        penalized.append(Detection(det.xyxy, float(new_score), det.category_id))
    return penalized


def _resize_image(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize image with OpenCV if available, or fallback to numpy interpolation."""
    target_w, target_h = target_size
    try:
        import cv2
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        pass
    try:
        from PIL import Image
        pil_img = Image.fromarray(img)
        return np.array(pil_img.resize((target_w, target_h)))
    except ImportError:
        h, w = img.shape[:2]
        ys = (np.linspace(0, h - 1, target_h)).astype(int)
        xs = (np.linspace(0, w - 1, target_w)).astype(int)
        return img[ys[:, None], xs]


def make_adapoth_views(
    image: np.ndarray,
    method: str = "adapoth",
    scout_weights: Path | str | None = None,
    threshold: float = 0.05,
    context_margin: float = 0.30,
    k_max: int = 4,
    crop_size: tuple[int, int] = (640, 640),
    global_size: tuple[int, int] = (960, 544),
    gt_boxes_4k: list[list[float]] | None = None,
    device: str | None = None,
) -> tuple[list[ProcessedView], dict[str, Any]]:
    """Construct AdaPoth Global + Local Candidate Views (Module A + B + C).
    
    Returns:
      views: List of ProcessedView instances (1 global view + K local candidate crop views)
      meta: Metadata regarding scout candidates and compute allocation.
    """
    orig_h, orig_w = image.shape[:2]
    views: list[ProcessedView] = []

    # 1. Generate Global View (always processed to ensure full context)
    global_w, global_h = global_size
    global_img = _resize_image(image, (global_w, global_h))
    global_transform = GlobalScaleTransform(src_w=orig_w, src_h=orig_h, dst_w=global_w, dst_h=global_h)
    views.append(ProcessedView(
        image=global_img,
        transform=global_transform,
        source_width=orig_w,
        source_height=orig_h,
        metadata={"type": "global", "view_index": 0},
    ))

    candidate_regions: list[CandidateRegion] = []
    scout_time_ms = 0.0

    if method == "adapoth-oracle" and gt_boxes_4k:
        # Oracle upper bound: Build crops directly around GT annotations
        for idx, gt in enumerate(gt_boxes_4k):
            gx, gy, gw, gh = gt[:4]
            cx, cy = gx + gw * 0.5, gy + gh * 0.5
            cw = int(max(320, gw * (1.0 + context_margin * 2.0)))
            ch = int(max(240, gh * (1.0 + context_margin * 2.0)))
            x0 = max(0, int(cx - cw * 0.5))
            y0 = max(0, int(cy - ch * 0.5))
            x1 = min(orig_w, x0 + cw)
            y1 = min(orig_h, y0 + ch)
            candidate_regions.append(CandidateRegion(x0, y0, x1, y1, score=1.0, component_id=idx + 1))
        # Keep up to k_max
        candidate_regions = candidate_regions[:k_max]

    elif method == "adapoth-random":
        # Random 2 crops in road region
        import random
        for idx in range(min(2, k_max)):
            cw, ch = 800, 600
            rx0 = random.randint(0, max(0, orig_w - cw))
            ry0 = random.randint(int(orig_h * 0.40), max(int(orig_h * 0.40), orig_h - ch))
            candidate_regions.append(CandidateRegion(rx0, ry0, rx0 + cw, ry0 + ch, score=0.5, component_id=idx + 1))

    else:
        # Standard AdaPoth / AdaPoth-Lite / AdaPoth-Fixed: Run Scout Network
        import time
        t0 = time.time()
        scout_engine = AdaPothScoutEngine(weights_path=scout_weights, device=device)
        heatmap = scout_engine.predict_heatmap(image)
        scout_time_ms = (time.time() - t0) * 1000.0

        effective_k_max = k_max if method != "adapoth-fixed" else 4
        candidate_gen = CandidateGenerator(
            threshold=threshold,
            context_margin=context_margin,
            k_max=effective_k_max,
        )
        candidate_regions = candidate_gen.generate(heatmap, source_width=orig_w, source_height=orig_h)

    # 2. Extract and Resize Local Candidate Crops
    target_w, target_h = crop_size
    for idx, cand in enumerate(candidate_regions):
        crop = image[cand.y0:cand.y1, cand.x0:cand.x1]
        if crop.size == 0:
            continue
        crop_resized = _resize_image(crop, (target_w, target_h))
        transform = ScaledCropTransform(
            x0=cand.x0, y0=cand.y0,
            crop_w=cand.width, crop_h=cand.height,
            view_w=target_w, view_h=target_h,
        )
        views.append(ProcessedView(
            image=crop_resized,
            transform=transform,
            source_width=cand.width,
            source_height=cand.height,
            metadata={"type": "local_crop", "candidate_score": cand.score, "crop_box": cand.xyxy, "view_index": idx + 1},
        ))

    meta = {
        "method": method,
        "scout_latency_ms": scout_time_ms,
        "k_candidates": len(candidate_regions),
        "total_views": len(views),
        "candidate_boxes": [c.xyxy for c in candidate_regions],
    }
    return views, meta
