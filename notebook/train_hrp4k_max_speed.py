# AdaPoth-Lite 96GB VRAM Max Speed Strategy Training Script for HRP4K Dataset
import os
import sys
import json
import time
import math
import glob
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

@dataclass
class AdaPothConfig:
    DATA_DIR: str = field(default_factory=lambda: os.getenv("DATA_DIR", "./HRP4K"))
    RESULTS_DIR: str = field(default_factory=lambda: os.getenv("RESULTS_DIR", "./results"))
    
    DEBUG_MINIMAL_DATA: bool = False
    NUM_DEBUG_SAMPLES: int = 20
    
    ORIGINAL_WIDTH: int = 3840
    ORIGINAL_HEIGHT: int = 2160
    
    # Scout Hyperparameters
    SCOUT_INPUT_W: int = 960
    SCOUT_INPUT_H: int = 540
    SCOUT_STRIDE: int = 16
    HEATMAP_ALPHA: float = 0.35
    HEATMAP_BETA: float = 0.50
    SCOUT_TAU: float = 0.30
    
    # Crop Extractor (Max Speed Strategy: 384 x 256)
    K_MAX: int = 4
    LOCAL_CROP_W: int = 384
    LOCAL_CROP_H: int = 256
    CONTEXT_MARGIN: float = 0.20
    REGION_NMS_IOU: float = 0.35
    
    # Shared Detector (Max Speed Strategy: Local Detector 384 x 256)
    GLOBAL_INPUT_W: int = 960
    GLOBAL_INPUT_H: int = 544
    LOCAL_DETECTOR_W: int = 384
    LOCAL_DETECTOR_H: int = 256
    YOLO_MODEL_NAME: str = "yolo11n.pt"
    
    # Max Speed Strategy High Throughput Batch Sizes & Epochs
    BATCH_SIZE_SCOUT: int = 256
    BATCH_SIZE_DETECTOR: int = 128
    EPOCHS_SCOUT: int = 10
    EPOCHS_DETECTOR: int = 10
    PATIENCE_SCOUT: int = 5
    PATIENCE_DETECTOR: int = 5
    WARMUP_EPOCHS: int = 2
    LR_SCOUT: float = 1e-3
    LR_DETECTOR: float = 5e-4
    WEIGHT_DECAY: float = 1e-4
    NUM_WORKERS: int = 16
    PREFETCH_FACTOR: int = 4
    PERSISTENT_WORKERS: bool = True
    PIN_MEMORY: bool = True
    USE_AMP: bool = True
    SEED: int = 42
    
    # Fusion & Thresholds (CONF_THRESH = 0.10)
    CONF_THRESH: float = 0.10
    NMS_IOU_THRESH: float = 0.45
    GLOBAL_SCORE_WEIGHT: float = 1.0
    LOCAL_SCORE_WEIGHT: float = 1.1
    BOUNDARY_PENALTY: float = 0.85
    
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

config = AdaPothConfig()

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

setup_seed(config.SEED)

class HRP4KDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", config: AdaPothConfig = None):
        self.data_dir = data_dir
        self.split = split
        self.config = config or AdaPothConfig()
        
        # Check candidate locations for split json and images
        base_paths = [data_dir, os.path.join(data_dir, "HRP4K"), "../HRP4K", "../HRP4K/HRP4K", "/marimo/HRP4K/HRP4K"]
        
        candidate_jsons = []
        for bp in base_paths:
            candidate_jsons.append(os.path.join(bp, f"{split}.json"))
            
        if split in ["valid", "val", "test"]:
            for bp in base_paths:
                candidate_jsons.extend([
                    os.path.join(bp, "test.json"),
                    os.path.join(bp, "valid.json"),
                ])
            
        json_path = None
        for p in candidate_jsons:
            if os.path.exists(p):
                json_path = p
                break
                
        if not json_path:
            raise FileNotFoundError(f"[ERROR] Annotation file missing for split '{split}' in {data_dir}. Fallback disabled!")
            
        self.json_path = json_path
        actual_split_name = Path(json_path).stem
        
        candidate_img_dirs = []
        for bp in base_paths:
            candidate_img_dirs.extend([
                os.path.join(bp, actual_split_name, "images"),
                os.path.join(bp, split, "images"),
            ])
        
        img_dir = None
        for p in candidate_img_dirs:
            if os.path.exists(p):
                img_dir = p
                break
                
        if not img_dir:
            raise FileNotFoundError(f"[ERROR] Image directory missing for split '{split}' in {data_dir}. Fallback disabled!")
            
        self.img_dir = img_dir
        self.coco = COCO(self.json_path)
        all_ids = list(self.coco.imgs.keys())
        
        self._img_cache = {}
        for f in os.listdir(self.img_dir):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                self._img_cache[f] = os.path.join(self.img_dir, f)
        
        valid_img_ids = []
        for img_id in all_ids:
            fn = self.coco.imgs[img_id]["file_name"]
            base_fn = os.path.basename(fn)
            img_p = self._img_cache.get(base_fn) or os.path.join(self.img_dir, base_fn)
            if os.path.exists(img_p) and os.path.getsize(img_p) > 0:
                valid_img_ids.append(img_id)
                self._img_cache[img_id] = img_p
                
        if len(valid_img_ids) == 0:
            raise RuntimeError(f"[ERROR] Zero valid images found for split '{split}' in {self.img_dir}. Fallback disabled!")
            
        self.img_ids = valid_img_ids
        print(f"[DATASET] [{split} -> {actual_split_name}] Loaded {len(self.img_ids)} valid images physically present on disk.")

    def __len__(self):
        return len(self.img_ids)

    def generate_gaussian_heatmap(self, bboxes_orig: List[List[float]], orig_w: int, orig_h: int) -> torch.Tensor:
        grid_w = self.config.SCOUT_INPUT_W // self.config.SCOUT_STRIDE
        grid_h = self.config.SCOUT_INPUT_H // self.config.SCOUT_STRIDE
        heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        scale_x = grid_w / float(orig_w)
        scale_y = grid_h / float(orig_h)
        
        for bbox in bboxes_orig:
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            
            cx_grid = (x + w / 2.0) * scale_x
            cy_grid = (y + h / 2.0) * scale_y
            w_grid = max(1.0, w * scale_x)
            h_grid = max(1.0, h * scale_y)
            
            sigma_x = max(0.5, self.config.HEATMAP_ALPHA * w_grid)
            sigma_y = max(0.5, self.config.HEATMAP_BETA * h_grid)
            
            radius_x = int(3 * sigma_x)
            radius_y = int(3 * sigma_y)
            
            x0 = max(0, int(cx_grid - radius_x))
            x1 = min(grid_w, int(cx_grid + radius_x + 1))
            y0 = max(0, int(cy_grid - radius_y))
            y1 = min(grid_h, int(cy_grid + radius_y + 1))
            
            if x1 <= x0 or y1 <= y0:
                continue
                
            grid_x, grid_y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            gaussian = np.exp(-(((grid_x - cx_grid) ** 2) / (2 * sigma_x ** 2) + ((grid_y - cy_grid) ** 2) / (2 * sigma_y ** 2)))
            heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)
            
            cx_int = int(np.clip(cx_grid, 0, grid_w - 1))
            cy_int = int(np.clip(cy_grid, 0, grid_h - 1))
            heatmap[cy_int, cx_int] = 1.0
            
        return torch.from_numpy(heatmap).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        file_name = img_info["file_name"]
        img_path = self._img_cache.get(img_id)
        
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"[ERROR] Image file '{file_name}' not found for img_id {img_id}. Fallback disabled!")
            
        image_cv = cv2.imread(img_path)
        if image_cv is None:
            raise RuntimeError(f"[ERROR] cv2 failed to read image file '{img_path}'. Fallback disabled!")
            
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_cv.shape[:2]
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        bboxes = []
        labels = []
        for ann in anns:
            bbox = ann["bbox"]
            if bbox[2] > 0 and bbox[3] > 0:
                bboxes.append(bbox)
                labels.append(ann.get("category_id", 0))
                
        scout_img = cv2.resize(image_cv, (self.config.SCOUT_INPUT_W, self.config.SCOUT_INPUT_H))
        scout_tensor = torch.from_numpy(scout_img).permute(2, 0, 1).float() / 255.0
        scout_normalized = TF.normalize(scout_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        
        gt_heatmap = self.generate_gaussian_heatmap(bboxes, orig_w, orig_h)
        image_tensor = torch.from_numpy(image_cv).permute(2, 0, 1)
        
        boxes_pascal = []
        for b in bboxes:
            boxes_pascal.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
            
        if len(boxes_pascal) == 0:
            boxes_pascal = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_pascal = torch.tensor(boxes_pascal, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            
        return {
            "image_id": img_id,
            "file_name": file_name,
            "image_4k": image_tensor,
            "scout_image": scout_normalized,
            "gt_heatmap": gt_heatmap,
            "boxes_4k": boxes_pascal,
            "labels": labels_tensor,
            "orig_size": (orig_w, orig_h)
        }

def collate_fn(batch):
    return {
        "image_id": [item["image_id"] for item in batch],
        "file_name": [item["file_name"] for item in batch],
        "image_4k": [item["image_4k"] for item in batch],
        "scout_image": torch.stack([item["scout_image"] for item in batch]),
        "gt_heatmap": torch.stack([item["gt_heatmap"] for item in batch]),
        "boxes_4k": [item["boxes_4k"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "orig_size": [item["orig_size"] for item in batch]
    }

class MobileNetV3Scout(nn.Module):
    def __init__(self, target_h: int = 33, target_w: int = 60, pretrained: bool = True):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        
        self.head = nn.Sequential(
            nn.Conv2d(576, 576, kernel_size=3, padding=1, groups=576, bias=False),
            nn.BatchNorm2d(576),
            nn.Hardswish(inplace=True),
            nn.Conv2d(576, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        heatmap = self.head(feat)
        if heatmap.shape[-2] != self.target_h or heatmap.shape[-1] != self.target_w:
            heatmap = F.interpolate(heatmap, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False)
        return heatmap

class FocalCoverageLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, lambda_cov: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_cov = lambda_cov

    def forward(self, pred_heatmap: torch.Tensor, gt_heatmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_mask = gt_heatmap.ge(0.95).float()
        neg_mask = gt_heatmap.lt(0.95).float()
        neg_weights = torch.pow(1.0 - gt_heatmap, self.beta)
        
        pos_loss = torch.log(pred_heatmap + 1e-6) * torch.pow(1.0 - pred_heatmap, self.alpha) * pos_mask
        neg_loss = torch.log(1.0 - pred_heatmap + 1e-6) * torch.pow(pred_heatmap, self.alpha) * neg_weights * neg_mask
        
        num_pos = max(1.0, float(pos_mask.sum().item()))
        focal_loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        pos_coverage = (pos_mask * (1.0 - pred_heatmap)).pow(2).sum() / num_pos
        
        total_loss = focal_loss + self.lambda_cov * pos_coverage
        return total_loss, focal_loss, pos_coverage

class DynamicTopKScout:
    def __init__(self, config: AdaPothConfig):
        self.config = config

    def extract_crops(
        self,
        heatmap_tensor: torch.Tensor,
        image_4k_tensor: torch.Tensor,
        orig_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]]]:
        heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
        grid_h, grid_w = heatmap_np.shape
        
        _, h_img, w_img = image_4k_tensor.shape
        orig_w = orig_size[0] if orig_size else w_img
        orig_h = orig_size[1] if orig_size else h_img
        
        binary_map = (heatmap_np >= self.config.SCOUT_TAU).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map)
        
        candidates = []
        scale_x = orig_w / float(grid_w)
        scale_y = orig_h / float(grid_h)
        
        for i in range(1, num_labels):
            x_g, y_g, w_g, h_g, area = stats[i]
            if area < 2:
                continue
                
            mask = (labels == i)
            region_score = float(heatmap_np[mask].max())
            
            x_min = x_g * scale_x
            y_min = y_g * scale_y
            x_max = (x_g + w_g) * scale_x
            y_max = (y_g + h_g) * scale_y
            
            w_4k = x_max - x_min
            h_4k = y_max - y_min
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            
            target_w = max(float(self.config.LOCAL_CROP_W), w_4k * (1.0 + self.config.CONTEXT_MARGIN))
            target_h = max(float(self.config.LOCAL_CROP_H), h_4k * (1.0 + self.config.CONTEXT_MARGIN))
            
            c_xmin = int(round(cx - target_w / 2.0))
            c_ymin = int(round(cy - target_h / 2.0))
            
            c_xmin = max(0, min(int(orig_w - target_w), c_xmin)) if orig_w >= target_w else 0
            c_ymin = max(0, min(int(orig_h - target_h), c_ymin)) if orig_h >= target_h else 0
            c_xmax = min(orig_w, int(c_xmin + target_w))
            c_ymax = min(orig_h, int(c_ymin + target_h))
            
            candidates.append(([c_xmin, c_ymin, c_xmax, c_ymax], region_score))
            
        if len(candidates) == 0:
            max_idx = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            cy_4k = (max_idx[0] + 0.5) * scale_y
            cx_4k = (max_idx[1] + 0.5) * scale_x
            
            target_w = float(self.config.LOCAL_CROP_W)
            target_h = float(self.config.LOCAL_CROP_H)
            
            c_xmin = max(0, min(int(orig_w - target_w), int(cx_4k - target_w / 2.0)))
            c_ymin = max(0, min(int(orig_h - target_h), int(cy_4k - target_h / 2.0)))
            c_xmax = min(orig_w, int(c_xmin + target_w))
            c_ymax = min(orig_h, int(c_ymin + target_h))
            
            candidates.append(([c_xmin, c_ymin, c_xmax, c_ymax], float(heatmap_np[max_idx])))
            
        candidates.sort(key=lambda item: item[1], reverse=True)
        selected_boxes = []
        
        for box, score in candidates:
            keep = True
            for sel_box in selected_boxes:
                ix1, iy1 = max(box[0], sel_box[0]), max(box[1], sel_box[1])
                ix2, iy2 = min(box[2], sel_box[2]), min(box[3], sel_box[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area1 = (box[2] - box[0]) * (box[3] - box[1])
                area2 = (sel_box[2] - sel_box[0]) * (sel_box[3] - sel_box[1])
                union = area1 + area2 - inter
                iou = inter / float(union + 1e-6)
                
                if iou > self.config.REGION_NMS_IOU:
                    keep = False
                    break
            if keep:
                selected_boxes.append(box)
                if len(selected_boxes) >= self.config.K_MAX:
                    break
                    
        crops = []
        final_crop_boxes = []
        
        for box in selected_boxes:
            x1, y1, x2, y2 = box
            crop_t = image_4k_tensor[:, y1:y2, x1:x2]
            if crop_t.dtype == torch.uint8:
                crop_t = crop_t.float() / 255.0
            
            if crop_t.shape[1] != self.config.LOCAL_CROP_H or crop_t.shape[2] != self.config.LOCAL_CROP_W:
                crop_t = F.interpolate(crop_t.unsqueeze(0), size=(self.config.LOCAL_CROP_H, self.config.LOCAL_CROP_W), mode="bilinear", align_corners=False).squeeze(0)
                
            crops.append(crop_t)
            final_crop_boxes.append((x1, y1, x2, y2))
            
        return crops, final_crop_boxes

class SharedGlobalLocalDetector(nn.Module):
    def __init__(self, model_name: str = "yolo11n.pt", config: AdaPothConfig = None):
        super().__init__()
        self.config = config or AdaPothConfig()
        self.__dict__["_yolo"] = YOLO(model_name)
        self.py_model = self.__dict__["_yolo"].model
        self.unfreeze()
        self.py_model.args = get_cfg()
        self.compute_loss = v8DetectionLoss(self.py_model)

    def unfreeze(self):
        for p in self.py_model.parameters():
            p.requires_grad = True

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.py_model.to(*args, **kwargs)
        if hasattr(self, "compute_loss"):
            try:
                device = next(self.py_model.parameters()).device
                if hasattr(self.compute_loss, "proj") and self.compute_loss.proj is not None:
                    self.compute_loss.proj = self.compute_loss.proj.to(device)
                if hasattr(self.compute_loss, "device"):
                    self.compute_loss.device = device
            except Exception:
                pass
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.py_model.train(mode)
        if mode:
            self.unfreeze()
        return self

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        if self.training:
            if isinstance(images, list):
                images_tensor = torch.stack(images).to(self.config.DEVICE)
            else:
                images_tensor = images.to(self.config.DEVICE)
                
            preds = self.py_model(images_tensor)
            
            batch_idx = []
            cls_list = []
            bboxes_list = []
            
            img_h, img_w = images_tensor.shape[2], images_tensor.shape[3]
            
            for b_idx, target in enumerate(targets):
                boxes = target["boxes"]
                labels = target["labels"]
                
                if len(boxes) > 0:
                    bw = (boxes[:, 2] - boxes[:, 0]) / float(img_w)
                    bh = (boxes[:, 3] - boxes[:, 1]) / float(img_h)
                    bcx = (boxes[:, 0] + boxes[:, 2]) / 2.0 / float(img_w)
                    bcy = (boxes[:, 1] + boxes[:, 3]) / 2.0 / float(img_h)
                    
                    xywh = torch.stack([bcx, bcy, bw, bh], dim=-1)
                    for k in range(len(boxes)):
                        batch_idx.append(b_idx)
                        cls_list.append(labels[k].item() if labels[k].item() >= 0 else 0)
                        bboxes_list.append(xywh[k])
                        
            if len(batch_idx) > 0:
                yolo_batch = {
                    "batch_idx": torch.tensor(batch_idx, device=images_tensor.device),
                    "cls": torch.tensor(cls_list, device=images_tensor.device).unsqueeze(1),
                    "bboxes": torch.stack(bboxes_list, dim=0).to(images_tensor.device),
                    "imgsz": torch.tensor([img_h, img_w], device=images_tensor.device)
                }
            else:
                yolo_batch = {
                    "batch_idx": torch.zeros((0,), dtype=torch.int64, device=images_tensor.device),
                    "cls": torch.zeros((0, 1), dtype=torch.float32, device=images_tensor.device),
                    "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=images_tensor.device),
                    "imgsz": torch.tensor([img_h, img_w], device=images_tensor.device)
                }
                
            if hasattr(self.compute_loss, "proj") and self.compute_loss.proj is not None:
                if self.compute_loss.proj.device != images_tensor.device:
                    self.compute_loss.proj = self.compute_loss.proj.to(images_tensor.device)
            if hasattr(self.compute_loss, "device"):
                self.compute_loss.device = images_tensor.device
                
            loss, loss_items = self.compute_loss(preds, yolo_batch)
            return {"total_loss": loss.sum(), "box_loss": loss_items.get("box_loss", 0), "cls_loss": loss_items.get("cls_loss", 0)}
        else:
            outputs = []
            for img_t in images:
                img_np = (img_t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                results = self.__dict__["_yolo"].predict(img_np, verbose=False, conf=self.config.CONF_THRESH, iou=self.config.NMS_IOU_THRESH)
                res = results[0]
                boxes = res.boxes.xyxy.cpu()
                scores = res.boxes.conf.cpu()
                labels = res.boxes.cls.cpu().long()
                outputs.append({"boxes": boxes, "scores": scores, "labels": labels})
            return outputs

class AdaPothLitePipeline(nn.Module):
    def __init__(self, scout: MobileNetV3Scout, detector: SharedGlobalLocalDetector, config: AdaPothConfig):
        super().__init__()
        self.scout = scout
        self.detector = detector
        self.config = config
        self.crop_extractor = DynamicTopKScout(config)

    def forward_single_image(self, image_4k: torch.Tensor, scout_img: torch.Tensor, orig_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        self.scout.eval()
        self.detector.eval()
        
        if image_4k.dtype == torch.uint8:
            image_4k = image_4k.float() / 255.0
            
        orig_w = orig_size[0] if orig_size else image_4k.shape[-1]
        orig_h = orig_size[1] if orig_size else image_4k.shape[-2]
        
        with torch.no_grad():
            scout_input = scout_img.unsqueeze(0).to(self.config.DEVICE)
            pred_heatmap = self.scout(scout_input)
            
            crops, crop_boxes = self.crop_extractor.extract_crops(pred_heatmap.squeeze(0), image_4k, (orig_w, orig_h))
            
            global_img = F.interpolate(image_4k.unsqueeze(0), size=(self.config.GLOBAL_INPUT_H, self.config.GLOBAL_INPUT_W), mode="bilinear", align_corners=False).squeeze(0)
            global_out = self.detector([global_img.to(self.config.DEVICE)])[0]
            
            scale_x_g = orig_w / float(self.config.GLOBAL_INPUT_W)
            scale_y_g = orig_h / float(self.config.GLOBAL_INPUT_H)
            
            g_boxes = global_out["boxes"].cpu()
            g_scores = torch.clamp(global_out["scores"].cpu() * self.config.GLOBAL_SCORE_WEIGHT, 0.0, 1.0)
            g_labels = global_out["labels"].cpu()
            
            g_boxes_4k = torch.zeros_like(g_boxes)
            if len(g_boxes) > 0:
                g_boxes_4k[:, 0] = g_boxes[:, 0] * scale_x_g
                g_boxes_4k[:, 1] = g_boxes[:, 1] * scale_y_g
                g_boxes_4k[:, 2] = g_boxes[:, 2] * scale_x_g
                g_boxes_4k[:, 3] = g_boxes[:, 3] * scale_y_g
                
            l_boxes_4k_list = []
            l_scores_list = []
            l_labels_list = []
            
            if len(crops) > 0:
                crop_batch = [c.to(self.config.DEVICE) for c in crops]
                local_outs = self.detector(crop_batch)
                
                for loc_out, (cx1, cy1, cx2, cy2) in zip(local_outs, crop_boxes):
                    l_b = loc_out["boxes"].cpu()
                    l_s = loc_out["scores"].cpu()
                    l_lbl = loc_out["labels"].cpu()
                    
                    if len(l_b) > 0:
                        boundary_margin = 5
                        on_boundary = (l_b[:, 0] <= boundary_margin) | (l_b[:, 1] <= boundary_margin) | \
                                      (l_b[:, 2] >= self.config.LOCAL_DETECTOR_W - boundary_margin) | \
                                      (l_b[:, 3] >= self.config.LOCAL_DETECTOR_H - boundary_margin)
                        
                        penalty_weights = torch.where(on_boundary, float(self.config.BOUNDARY_PENALTY), 1.0)
                        calibrated_scores = torch.clamp(l_s * self.config.LOCAL_SCORE_WEIGHT * penalty_weights, 0.0, 1.0)
                        
                        scale_x_l = float(cx2 - cx1) / float(self.config.LOCAL_DETECTOR_W)
                        scale_y_l = float(cy2 - cy1) / float(self.config.LOCAL_DETECTOR_H)
                        
                        mapped_b = torch.zeros_like(l_b)
                        mapped_b[:, 0] = l_b[:, 0] * scale_x_l + cx1
                        mapped_b[:, 1] = l_b[:, 1] * scale_y_l + cy1
                        mapped_b[:, 2] = l_b[:, 2] * scale_x_l + cx1
                        mapped_b[:, 3] = l_b[:, 3] * scale_y_l + cy1
                        
                        l_boxes_4k_list.append(mapped_b)
                        l_scores_list.append(calibrated_scores)
                        l_labels_list.append(l_lbl)
                        
            all_boxes = [g_boxes_4k] + l_boxes_4k_list
            all_scores = [g_scores] + l_scores_list
            all_labels = [g_labels] + l_labels_list
            
            all_boxes_t = torch.cat(all_boxes, dim=0) if len(all_boxes) > 0 else torch.zeros((0, 4))
            all_scores_t = torch.cat(all_scores, dim=0) if len(all_scores) > 0 else torch.zeros((0,))
            all_labels_t = torch.cat(all_labels, dim=0) if len(all_labels) > 0 else torch.zeros((0,), dtype=torch.int64)
            
            keep_mask = all_scores_t >= self.config.CONF_THRESH
            f_boxes = all_boxes_t[keep_mask]
            f_scores = all_scores_t[keep_mask]
            f_labels = all_labels_t[keep_mask]
            
            if len(f_boxes) > 0:
                keep_nms = torchvision.ops.batched_nms(f_boxes, f_scores, f_labels, self.config.NMS_IOU_THRESH)
                final_boxes = f_boxes[keep_nms]
                final_scores = f_scores[keep_nms]
                final_labels = f_labels[keep_nms]
            else:
                final_boxes = torch.zeros((0, 4))
                final_scores = torch.zeros((0,))
                final_labels = torch.zeros((0,), dtype=torch.int64)
                
            return {
                "pred_heatmap": pred_heatmap.squeeze(0),
                "crop_boxes": crop_boxes,
                "num_crops": len(crops),
                "final_boxes": final_boxes,
                "final_scores": final_scores,
                "final_labels": final_labels
            }

class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max", delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            return True
        
        if self.mode == "max":
            improved = val_score > (self.best_score + self.delta)
        else:
            improved = val_score < (self.best_score - self.delta)
            
        if improved:
            self.best_score = val_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def update_status_file(status_path: str, data: Dict[str, Any]):
    try:
        with open(status_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Failed to update status file: {e}")

def set_epoch_lr(optimizer: torch.optim.Optimizer, initial_lr: float, epoch: int, warmup_epochs: int = 2):
    if epoch <= warmup_epochs:
        warmup_factor = float(epoch) / float(max(1, warmup_epochs))
        current_lr = initial_lr * max(0.1, warmup_factor)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

def train_scout_epoch(
    scout: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: FocalCoverageLoss,
    device: str,
    epoch: int = 1,
    total_epochs: int = 1,
    scaler: Optional[torch.amp.GradScaler] = None
) -> Tuple[float, float, float]:
    scout.train()
    total_loss = 0.0
    total_focal = 0.0
    total_coverage = 0.0
    use_amp = (device == "cuda")
    
    pbar = tqdm(loader, desc=f"Scout Ep [{epoch}/{total_epochs}]", leave=False)
    for batch in pbar:
        scout_imgs = batch["scout_image"].to(device)
        gt_heatmaps = batch["gt_heatmap"].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            pred_heatmaps = scout(scout_imgs)
            loss, f_loss, c_loss = criterion(pred_heatmaps, gt_heatmaps)
            
        if scaler and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        bs = scout_imgs.size(0)
        total_loss += loss.item() * bs
        total_focal += f_loss.item() * bs
        total_coverage += c_loss.item() * bs
        
        current_lr = optimizer.param_groups[0]["lr"]
        gpu_mem = f"{torch.cuda.memory_reserved() / 1e9:.2f}GB" if torch.cuda.is_available() else "N/A"
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Focal": f"{f_loss.item():.4f}",
            "Cov": f"{c_loss.item():.4f}",
            "LR": f"{current_lr:.1e}",
            "GPU": gpu_mem
        })
        
    num_samples = len(loader.dataset)
    return (
        total_loss / max(1, num_samples),
        total_focal / max(1, num_samples),
        total_coverage / max(1, num_samples)
    )

def train_detector_epoch(
    detector: SharedGlobalLocalDetector,
    scout: MobileNetV3Scout,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: AdaPothConfig,
    epoch: int = 1,
    total_epochs: int = 1,
    scaler: Optional[torch.amp.GradScaler] = None
) -> Tuple[float, float, float]:
    detector.train()
    scout.eval()
    total_loss = 0.0
    total_g_loss = 0.0
    total_l_loss = 0.0
    crop_extractor = DynamicTopKScout(config)
    use_amp = (config.DEVICE == "cuda")
    
    pbar = tqdm(loader, desc=f"Detector Ep [{epoch}/{total_epochs}]", leave=False)
    for batch in pbar:
        imgs_4k = batch["image_4k"]
        scout_imgs = batch["scout_image"].to(config.DEVICE)
        gt_boxes_list = batch["boxes_4k"]
        gt_labels_list = batch["labels"]
        gt_heatmaps = batch["gt_heatmap"]
        orig_sizes = batch["orig_size"]
        
        with torch.no_grad():
            pred_heatmaps = scout(scout_imgs)
            
        g_images = []
        g_targets = []
        l_images = []
        l_targets = []
        
        for i in range(len(imgs_4k)):
            img_4k = imgs_4k[i]
            if img_4k.dtype == torch.uint8:
                img_4k = img_4k.float() / 255.0
            boxes_4k = gt_boxes_list[i].to(config.DEVICE)
            labels = gt_labels_list[i].to(config.DEVICE)
            orig_w, orig_h = orig_sizes[i]
            
            g_img = F.interpolate(img_4k.unsqueeze(0), size=(config.GLOBAL_INPUT_H, config.GLOBAL_INPUT_W), mode="bilinear", align_corners=False).squeeze(0).to(config.DEVICE)
            
            scale_x_g = float(config.GLOBAL_INPUT_W) / orig_w
            scale_y_g = float(config.GLOBAL_INPUT_H) / orig_h
            
            g_boxes = torch.zeros_like(boxes_4k)
            if len(boxes_4k) > 0:
                g_boxes[:, 0] = boxes_4k[:, 0] * scale_x_g
                g_boxes[:, 1] = boxes_4k[:, 1] * scale_y_g
                g_boxes[:, 2] = boxes_4k[:, 2] * scale_x_g
                g_boxes[:, 3] = boxes_4k[:, 3] * scale_y_g
                
            g_images.append(g_img)
            g_targets.append({"boxes": g_boxes, "labels": labels})
            
            gt_crops, gt_crop_boxes = crop_extractor.extract_crops(gt_heatmaps[i], img_4k, (orig_w, orig_h))
            pred_crops, pred_crop_boxes = crop_extractor.extract_crops(pred_heatmaps[i], img_4k, (orig_w, orig_h))
            
            combined_crops = gt_crops + pred_crops
            combined_crop_boxes = gt_crop_boxes + pred_crop_boxes
            
            for crop_t, (cx1, cy1, cx2, cy2) in zip(combined_crops, combined_crop_boxes):
                crop_t = crop_t.to(config.DEVICE)
                scale_x_l = float(config.LOCAL_DETECTOR_W) / max(1.0, float(cx2 - cx1))
                scale_y_l = float(config.LOCAL_DETECTOR_H) / max(1.0, float(cy2 - cy1))
                
                c_boxes = []
                c_labels = []
                for b, l in zip(boxes_4k, labels):
                    bx1, by1, bx2, by2 = b.tolist()
                    ix1 = max(bx1, cx1)
                    iy1 = max(by1, cy1)
                    ix2 = min(bx2, cx2)
                    iy2 = min(by2, cy2)
                    if ix2 > ix1 and iy2 > iy1:
                        nbx1 = (ix1 - cx1) * scale_x_l
                        nby1 = (iy1 - cy1) * scale_y_l
                        nbx2 = (ix2 - cx1) * scale_x_l
                        nby2 = (iy2 - cy1) * scale_y_l
                        c_boxes.append([nbx1, nby1, nbx2, nby2])
                        c_labels.append(l.item())
                        
                if len(c_boxes) > 0:
                    c_boxes_t = torch.tensor(c_boxes, dtype=torch.float32, device=config.DEVICE)
                    c_labels_t = torch.tensor(c_labels, dtype=torch.int64, device=config.DEVICE)
                else:
                    c_boxes_t = torch.zeros((0, 4), dtype=torch.float32, device=config.DEVICE)
                    c_labels_t = torch.zeros((0,), dtype=torch.int64, device=config.DEVICE)
                    
                l_images.append(crop_t)
                l_targets.append({"boxes": c_boxes_t, "labels": c_labels_t})
                
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            g_out = detector(g_images, g_targets)
            g_loss = g_out["total_loss"]
            l_loss = torch.tensor(0.0, device=config.DEVICE)
            
            if len(l_images) > 0:
                l_out = detector(l_images, l_targets)
                l_loss = l_out["total_loss"]
                
            losses = g_loss + l_loss
                
        if scaler and use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
            
        bs = len(imgs_4k)
        total_loss += losses.item() * bs
        total_g_loss += g_loss.item() * bs
        total_l_loss += l_loss.item() * bs
        
        current_lr = optimizer.param_groups[0]["lr"]
        gpu_mem = f"{torch.cuda.memory_reserved() / 1e9:.2f}GB" if torch.cuda.is_available() else "N/A"
        pbar.set_postfix({
            "Loss": f"{losses.item():.4f}",
            "G_Loss": f"{g_loss.item():.4f}",
            "L_Loss": f"{l_loss.item():.4f}",
            "LR": f"{current_lr:.1e}",
            "GPU": gpu_mem
        })
        
    num_samples = len(loader.dataset)
    return (
        total_loss / max(1, num_samples),
        total_g_loss / max(1, num_samples),
        total_l_loss / max(1, num_samples)
    )

def evaluate_pipeline(pipeline: AdaPothLitePipeline, loader: DataLoader, config: AdaPothConfig) -> Dict[str, float]:
    total_gt_boxes = 0
    recalled_gt_boxes = 0
    total_crops = 0
    total_images = len(loader.dataset)
    
    coco_gt = loader.dataset.coco
    coco_predictions = []
    
    for batch in tqdm(loader, desc="Evaluating Pipeline", leave=False):
        for i in range(len(batch["image_4k"])):
            img_id = batch["image_id"][i]
            img_4k = batch["image_4k"][i]
            scout_img = batch["scout_image"][i]
            gt_boxes_4k = batch["boxes_4k"][i]
            orig_w, orig_h = batch["orig_size"][i]
            
            res = pipeline.forward_single_image(img_4k, scout_img, (orig_w, orig_h))
            
            crop_boxes = res["crop_boxes"]
            total_crops += res["num_crops"]
            
            pred_boxes = res["final_boxes"]
            pred_scores = res["final_scores"]
            pred_labels = res.get("final_labels", torch.zeros(len(pred_boxes), dtype=torch.int64))
            
            for gt_b in gt_boxes_4k:
                total_gt_boxes += 1
                gx1, gy1, gx2, gy2 = gt_b.tolist()
                covered = False
                for cx1, cy1, cx2, cy2 in crop_boxes:
                    if gx1 >= cx1 and gy1 >= cy1 and gx2 <= cx2 and gy2 <= cy2:
                        covered = True
                        break
                if covered:
                    recalled_gt_boxes += 1
                    
            for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels):
                px1, py1, px2, py2 = pb.tolist()
                pw = max(0.1, px2 - px1)
                ph = max(0.1, py2 - py1)
                cat_id = int(pl.item() if torch.is_tensor(pl) else pl)
                coco_predictions.append({
                    "image_id": int(img_id),
                    "category_id": cat_id,
                    "bbox": [round(px1, 2), round(py1, 2), round(pw, 2), round(ph, 2)],
                    "score": float(ps.item())
                })
                
    scout_recall = (recalled_gt_boxes / max(1, total_gt_boxes)) * 100.0
    avg_k = total_crops / float(max(1, total_images))
    
    coco_metrics = {
        "mAP_50_95": 0.0,
        "mAP_50": 0.0,
        "mAP_75": 0.0,
        "AP_small": 0.0,
        "AP_medium": 0.0,
        "AP_large": 0.0,
        "AR_100": 0.0
    }
    
    if len(coco_predictions) > 0:
        try:
            coco_dt = coco_gt.loadRes(coco_predictions)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.params.imgIds = list(loader.dataset.img_ids)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            stats = coco_eval.stats
            coco_metrics = {
                "mAP_50_95": round(float(stats[0]), 4),
                "mAP_50": round(float(stats[1]), 4),
                "mAP_75": round(float(stats[2]), 4),
                "AP_small": round(float(stats[3]), 4),
                "AP_medium": round(float(stats[4]), 4),
                "AP_large": round(float(stats[5]), 4),
                "AR_100": round(float(stats[8]), 4)
            }
        except Exception as e:
            print(f"[WARNING] COCOeval error: {e}")
            
    return {
        "scout_region_recall": round(scout_recall, 2),
        "average_crops_k": round(avg_k, 2),
        **coco_metrics
    }

def evaluate_scout_only(scout: nn.Module, loader: DataLoader, config: AdaPothConfig) -> Dict[str, float]:
    scout.eval()
    total_gt_boxes = 0
    recalled_gt_boxes = 0
    total_crops = 0
    total_images = len(loader.dataset)
    crop_extractor = DynamicTopKScout(config)
    use_amp = getattr(config, "USE_AMP", config.DEVICE == "cuda")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Scout", leave=False):
            for i in range(len(batch["image_4k"])):
                img_4k = batch["image_4k"][i]
                if img_4k.dtype == torch.uint8:
                    img_4k = img_4k.float() / 255.0
                scout_img = batch["scout_image"][i]
                gt_boxes_4k = batch["boxes_4k"][i]
                orig_w, orig_h = batch["orig_size"][i]
                
                scout_input = scout_img.unsqueeze(0).to(config.DEVICE)
                with torch.amp.autocast(device_type="cuda" if config.DEVICE == "cuda" else "cpu", enabled=use_amp):
                    pred_heatmap = scout(scout_input)
                    
                crops, crop_boxes = crop_extractor.extract_crops(pred_heatmap.squeeze(0), img_4k, (orig_w, orig_h))
                total_crops += len(crops)
                
                for gt_b in gt_boxes_4k:
                    total_gt_boxes += 1
                    gx1, gy1, gx2, gy2 = gt_b.tolist()
                    covered = False
                    for cx1, cy1, cx2, cy2 in crop_boxes:
                        if gx1 >= cx1 and gy1 >= cy1 and gx2 <= cx2 and gy2 <= cy2:
                            covered = True
                            break
                    if covered:
                        recalled_gt_boxes += 1
                        
    scout_recall = (recalled_gt_boxes / max(1, total_gt_boxes)) * 100.0
    avg_k = total_crops / float(max(1, total_images))
    return {
        "scout_region_recall": round(scout_recall, 2),
        "average_crops_k": round(avg_k, 2)
    }

def setup_results_directories(config: AdaPothConfig):
    ckpt_dir = os.path.join(config.RESULTS_DIR, "checkpoints")
    vis_dir = os.path.join(config.RESULTS_DIR, "visualizations")
    log_dir = os.path.join(config.RESULTS_DIR, "logs")
    rep_dir = os.path.join(config.RESULTS_DIR, "reports")
    
    for d in [ckpt_dir, vis_dir, log_dir, rep_dir]:
        os.makedirs(d, exist_ok=True)
    return ckpt_dir, vis_dir, log_dir, rep_dir

def setup_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("AdaPothLogger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    log_file = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter("%(message)s")
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    
    return logger

def plot_and_save_losses(scout_losses: List[float], detector_losses: List[float], save_path: str):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Scout Focal Loss', color=color)
    if len(scout_losses) > 0:
        ax1.plot(range(1, len(scout_losses) + 1), scout_losses, color=color, marker='o', label='Scout Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Shared Detector Loss', color=color)
    if len(detector_losses) > 0:
        ax2.plot(range(1, len(detector_losses) + 1), detector_losses, color=color, marker='s', label='Detector Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('AdaPoth-Lite Max Speed Strategy Training Losses')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_sample_visualization(
    image_4k_tensor: torch.Tensor,
    pred_heatmap: torch.Tensor,
    crop_boxes: List[Tuple[int, int, int, int]],
    final_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    save_path: str
):
    img_np = (image_4k_tensor.permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
    heatmap_np = pred_heatmap.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    img_gt = img_np.copy()
    for gb in gt_boxes:
        gx1, gy1, gx2, gy2 = map(int, gb.tolist())
        cv2.rectangle(img_gt, (gx1, gy1), (gx2, gy2), (0, 255, 0), 6)
    axes[0, 0].imshow(img_gt)
    axes[0, 0].set_title("Ground Truth Pothole Bounding Boxes (Green)")
    axes[0, 0].axis("off")
    
    heatmap_resized = cv2.resize(heatmap_np, (img_np.shape[1], img_np.shape[0]))
    norm_heatmap = (np.clip(heatmap_resized, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.4, 0)
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title("MobileNetV3 Scout 2D Heatmap Overlay")
    axes[0, 1].axis("off")
    
    img_crops = img_np.copy()
    for cx1, cy1, cx2, cy2 in crop_boxes:
        cv2.rectangle(img_crops, (cx1, cy1), (cx2, cy2), (255, 165, 0), 8)
    axes[1, 0].imshow(img_crops)
    axes[1, 0].set_title(f"Dynamic Top-K Scout Crops (K={len(crop_boxes)}) (Orange)")
    axes[1, 0].axis("off")
    
    img_det = img_np.copy()
    for pb in final_boxes:
        px1, py1, px2, py2 = map(int, pb.tolist())
        cv2.rectangle(img_det, (px1, py1), (px2, py2), (255, 0, 0), 6)
    axes[1, 1].imshow(img_det)
    axes[1, 1].set_title("AdaPoth-Lite Final Fused Detections (Red)")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    ckpt_dir, vis_dir, log_dir, rep_dir = setup_results_directories(config)
    logger = setup_logger(log_dir)
    status_path = os.path.join(log_dir, "status.json")
    
    status_data = {
        "stage": "INITIALIZING",
        "phase": "STARTUP",
        "start_time": time.time(),
        "current_epoch": 0,
        "total_epochs_scout": config.EPOCHS_SCOUT,
        "total_epochs_detector": config.EPOCHS_DETECTOR,
        "scout_losses": [],
        "detector_losses": [],
        "val_history": [],
        "latest_metrics": {},
        "is_finished": False
    }
    update_status_file(status_path, status_data)
    
    logger.info("=" * 75)
    logger.info("      AdaPoth-Lite: High-Resolution 4K Pothole Pipeline (HRP4K)")
    logger.info("      Max Speed Strategy (Crop 384x256 | Scout BS 256 | Detector BS 128)")
    logger.info("=" * 75)
    logger.info(f"[ENV] PyTorch: {torch.__version__} | Torchvision: {torchvision.__version__} | Device: {config.DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"[GPU] Device: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("[GPU] No CUDA device detected, running on CPU.")
        
    logger.info(f"[CONFIG] Data Dir: {config.DATA_DIR} | Results Dir: {config.RESULTS_DIR}")
    logger.info(f"[CONFIG] Scout BS: {config.BATCH_SIZE_SCOUT} | Detector BS: {config.BATCH_SIZE_DETECTOR} | Num Workers: {config.NUM_WORKERS}")
    logger.info(f"[CONFIG] Crop Size: {config.LOCAL_CROP_W}x{config.LOCAL_CROP_H} | CONF_THRESH: {config.CONF_THRESH}")
    
    logger.info("[STEP 1/5] Loading Datasets...")
    train_dataset = HRP4KDataset(config.DATA_DIR, split="train", config=config)
    valid_dataset = HRP4KDataset(config.DATA_DIR, split="test", config=config)
    
    train_loader_scout = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE_SCOUT,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False,
        pin_memory=config.PIN_MEMORY
    )
    
    train_loader_detector = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE_DETECTOR,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False,
        pin_memory=config.PIN_MEMORY
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False,
        pin_memory=config.PIN_MEMORY
    )
    
    logger.info(f"[DATASET] Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")
    
    logger.info("[STEP 2/5] Building AdaPoth-Lite Architecture with Shared YOLO11n Detector...")
    scout_model = MobileNetV3Scout(pretrained=True).to(config.DEVICE)
    detector_model = SharedGlobalLocalDetector(model_name=config.YOLO_MODEL_NAME, config=config).to(config.DEVICE)
    pipeline = AdaPothLitePipeline(scout_model, detector_model, config).to(config.DEVICE)
    
    scout_optimizer = torch.optim.AdamW(scout_model.parameters(), lr=config.LR_SCOUT, weight_decay=config.WEIGHT_DECAY)
    scout_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(scout_optimizer, T_max=config.EPOCHS_SCOUT, eta_min=1e-5)
    
    detector_optimizer = torch.optim.AdamW(detector_model.parameters(), lr=config.LR_DETECTOR, weight_decay=config.WEIGHT_DECAY)
    detector_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(detector_optimizer, T_max=config.EPOCHS_DETECTOR, eta_min=1e-5)
    
    scout_criterion = FocalCoverageLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=config.DEVICE == "cuda")
    
    scout_losses = []
    detector_losses = []
    val_history = []
    
    best_scout_recall = 0.0
    best_map_50_95 = -1.0
    
    scout_early_stop = EarlyStopping(patience=config.PATIENCE_SCOUT, mode="max")
    detector_early_stop = EarlyStopping(patience=config.PATIENCE_DETECTOR, mode="max")
    
    logger.info("[STEP 3/5] Starting Full Training Run...")
    start_time = time.time()
    
    status_data["stage"] = "TRAINING_SCOUT"
    status_data["phase"] = "SCOUT"
    update_status_file(status_path, status_data)
    
    logger.info("--- Phase A: Training MobileNetV3-Small Scout (10 Epochs) ---")
    for epoch in range(1, config.EPOCHS_SCOUT + 1):
        set_epoch_lr(scout_optimizer, config.LR_SCOUT, epoch, config.WARMUP_EPOCHS)
        ep_start = time.time()
        loss_s, focal_s, cov_s = train_scout_epoch(
            scout_model, train_loader_scout, scout_optimizer, scout_criterion, config.DEVICE,
            epoch=epoch, total_epochs=config.EPOCHS_SCOUT, scaler=scaler
        )
        scout_losses.append(loss_s)
        
        if epoch > config.WARMUP_EPOCHS:
            scout_scheduler.step()
            
        val_m = evaluate_scout_only(scout_model, valid_loader, config)
        current_recall = val_m["scout_region_recall"]
        ep_time = time.time() - ep_start
        current_lr = scout_optimizer.param_groups[0]["lr"]
        gpu_mem = f"{torch.cuda.memory_reserved() / 1e9:.2f}GB" if torch.cuda.is_available() else "N/A"
        
        logger.info(
            f"Scout Epoch [{epoch:02d}/{config.EPOCHS_SCOUT:02d}] - "
            f"Loss: {loss_s:.4f} (Focal: {focal_s:.4f}, Cov: {cov_s:.4f}) | "
            f"Scout Recall: {current_recall:.2f}% | "
            f"LR: {current_lr:.1e} | Time: {ep_time:.2f}s | GPU: {gpu_mem}"
        )
        
        status_data["current_epoch"] = epoch
        status_data["scout_losses"] = scout_losses
        status_data["latest_metrics"] = val_m
        update_status_file(status_path, status_data)
        
        improved = scout_early_stop(current_recall)
        if improved:
            best_scout_recall = current_recall
            torch.save({
                "epoch": epoch,
                "scout_state": scout_model.state_dict(),
                "optimizer_state": scout_optimizer.state_dict(),
                "best_scout_recall": best_scout_recall
            }, os.path.join(ckpt_dir, "scout_best.pt"))
            logger.info(f" -> [CHECKPOINT] Saved best Scout checkpoint (Recall: {best_scout_recall:.2f}%)")
            
        if scout_early_stop.early_stop:
            logger.info(f"[EARLY STOP] Scout training stopped early at epoch {epoch}.")
            break
            
    status_data["stage"] = "TRAINING_DETECTOR"
    status_data["phase"] = "DETECTOR"
    update_status_file(status_path, status_data)
    
    logger.info("--- Phase B: Training Shared YOLO11n Detector (Global + Local) (10 Epochs) ---")
    for epoch in range(1, config.EPOCHS_DETECTOR + 1):
        set_epoch_lr(detector_optimizer, config.LR_DETECTOR, epoch, config.WARMUP_EPOCHS)
        ep_start = time.time()
        loss_d, g_loss, l_loss = train_detector_epoch(
            detector_model, scout_model, train_loader_detector, detector_optimizer, config,
            epoch=epoch, total_epochs=config.EPOCHS_DETECTOR, scaler=scaler
        )
        detector_losses.append(loss_d)
        
        if epoch > config.WARMUP_EPOCHS:
            detector_scheduler.step()
            
        val_m = evaluate_pipeline(pipeline, valid_loader, config)
        val_history.append({"epoch": epoch, "loss_d": loss_d, "g_loss": g_loss, "l_loss": l_loss, **val_m})
        current_map = val_m["mAP_50_95"]
        ep_time = time.time() - ep_start
        current_lr = detector_optimizer.param_groups[0]["lr"]
        gpu_mem = f"{torch.cuda.memory_reserved() / 1e9:.2f}GB" if torch.cuda.is_available() else "N/A"
        
        logger.info(
            f"Detector Epoch [{epoch:02d}/{config.EPOCHS_DETECTOR:02d}] - "
            f"Loss: {loss_d:.4f} (Global: {g_loss:.4f}, Local: {l_loss:.4f}) | "
            f"mAP_50_95: {current_map:.4f} | mAP_50: {val_m['mAP_50']:.4f} | "
            f"LR: {current_lr:.1e} | Time: {ep_time:.2f}s | GPU: {gpu_mem}"
        )
        
        status_data["current_epoch"] = epoch
        status_data["detector_losses"] = detector_losses
        status_data["val_history"] = val_history
        status_data["latest_metrics"] = val_m
        update_status_file(status_path, status_data)
        
        improved = detector_early_stop(current_map)
        if improved:
            best_map_50_95 = current_map
            torch.save({
                "epoch": epoch,
                "detector_state": detector_model.state_dict(),
                "optimizer_state": detector_optimizer.state_dict(),
                "best_map_50_95": best_map_50_95
            }, os.path.join(ckpt_dir, "detector_best.pt"))
            
            torch.save({
                "epoch": epoch,
                "scout_state": scout_model.state_dict(),
                "detector_state": detector_model.state_dict(),
                "metrics": val_m
            }, os.path.join(ckpt_dir, "adapoth_best.pt"))
            logger.info(f" -> [CHECKPOINT] Saved best Pipeline checkpoint (mAP_50_95: {best_map_50_95:.4f})")
            
        if detector_early_stop.early_stop:
            logger.info(f"[EARLY STOP] Detector training stopped early at epoch {epoch}.")
            break
            
    training_time = time.time() - start_time
    logger.info(f"[INFO] Training completed in {training_time:.2f} seconds.")
    
    torch.save({
        "epoch_scout": len(scout_losses),
        "epoch_detector": len(detector_losses),
        "scout_state": scout_model.state_dict(),
        "detector_state": detector_model.state_dict(),
        "val_history": val_history
    }, os.path.join(ckpt_dir, "adapoth_last.pt"))
    logger.info(f"[CHECKPOINT] Saved last state checkpoint adapoth_last.pt")
    
    logger.info("[STEP 4/5] Final COCOeval Pipeline Evaluation...")
    metrics = evaluate_pipeline(pipeline, valid_loader, config)
    logger.info("=" * 55)
    logger.info("           OFFICIAL COCOEVAL METRICS REPORT")
    logger.info("=" * 55)
    for k, v in metrics.items():
        logger.info(f" - {k:<25}: {v}")
    logger.info("=" * 55)
    
    logger.info("[STEP 5/5] Exporting Plots and Reports...")
    loss_plot_path = os.path.join(vis_dir, "training_loss.png")
    plot_and_save_losses(scout_losses, detector_losses, loss_plot_path)
    
    sample_batch = next(iter(valid_loader))
    sample_img_4k = sample_batch["image_4k"][0]
    sample_scout_img = sample_batch["scout_image"][0]
    sample_gt_boxes = sample_batch["boxes_4k"][0]
    sample_orig_size = sample_batch["orig_size"][0]
    
    res = pipeline.forward_single_image(sample_img_4k, sample_scout_img, sample_orig_size)
    sample_vis_path = os.path.join(vis_dir, "sample_detection_overlay.png")
    save_sample_visualization(
        sample_img_4k,
        res["pred_heatmap"],
        res["crop_boxes"],
        res["final_boxes"],
        sample_gt_boxes,
        sample_vis_path
    )
    
    report_data = {
        "pipeline": "AdaPoth-Lite-YOLO11n-Max-Speed-Strategy",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0,
        "num_samples_evaluated": len(valid_dataset),
        "training_time_seconds": round(training_time, 2),
        "final_metrics": metrics,
        "val_history": val_history,
        "scout_losses": scout_losses,
        "detector_losses": detector_losses
    }
    
    json_report_path = os.path.join(rep_dir, "evaluation_report.json")
    with open(json_report_path, "w") as f:
        json.dump(report_data, f, indent=4)
        
    md_report_path = os.path.join(rep_dir, "evaluation_report.md")
    with open(md_report_path, "w") as f:
        f.write("# AdaPoth-Lite Evaluation Report (Max Speed Strategy)\n\n")
        f.write(f"- **GPU**: {report_data['gpu']} ({report_data['vram_gb']} GB VRAM)\n")
        f.write(f"- **Training Duration**: {training_time / 60:.2f} minutes\n\n")
        f.write("## Official COCOeval Metrics\n\n")
        f.write("| Metric | Value |\n|---|---:|\n")
        for k, v in metrics.items():
            f.write(f"| {k} | {v} |\n")
            
    status_data["stage"] = "COMPLETED"
    status_data["is_finished"] = True
    status_data["final_metrics"] = metrics
    update_status_file(status_path, status_data)
    
    logger.info(f"[REPORT] Saved reports to: {rep_dir}")
    logger.info("[SUCCESS] AdaPoth-Lite Max Speed Strategy Training completed successfully!")

if __name__ == "__main__":
    main()
