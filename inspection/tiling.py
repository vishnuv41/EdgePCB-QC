"""
Tiled High-Resolution Inference Module for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Enables inspection of large-format / high-resolution PCBs (e.g. PCB-Defect V2, 2.3MP - 30MP):
- Slices large PCB image into overlapping sub-windows (e.g. 640x640 with 20% overlap)
- Translates local tile bounding boxes to global board coordinate frame
- Applies Global Non-Maximum Suppression (NMS) to fuse detections across tile boundaries
- Prevents defect clipping and enables sub-millimeter defect detection
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def global_nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
    """Performs Non-Maximum Suppression across all tiled detections."""
    if not detections:
        return []

    # Sort descending by confidence
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept: List[Dict[str, Any]] = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        kept.append(best)

        remaining = []
        for d in sorted_dets:
            # Only suppress if of the same class or significant overlap
            same_class = d.get("class") == best.get("class")
            iou = compute_iou(best.get("bbox", []), d.get("bbox", []))
            if not (same_class and iou > iou_threshold):
                remaining.append(d)
        sorted_dets = remaining

    return kept


class TiledInferenceRunner:
    """
    Applies sliding-window tiled inference using an underlying VisionInspector.
    """

    def __init__(
        self,
        inspector,
        tile_size: int = 640,
        overlap: float = 0.20,
        iou_threshold: float = 0.45
    ):
        self.inspector = inspector
        self.tile_size = tile_size
        self.overlap = overlap
        self.iou_threshold = iou_threshold

    def slice_image(self, img: np.ndarray) -> List[Tuple[np.ndarray, int, int]]:
        """Slices full board image into overlapping tiles."""
        h, w = img.shape[:2]
        stride = int(self.tile_size * (1.0 - self.overlap))
        tiles = []

        y = 0
        while y < h:
            x = 0
            while x < w:
                # Clamp tile boundaries
                x_end = min(x + self.tile_size, w)
                y_end = min(y + self.tile_size, h)
                x_start = max(0, x_end - self.tile_size)
                y_start = max(0, y_end - self.tile_size)

                tile = img[y_start:y_end, x_start:x_end]
                tiles.append((tile, x_start, y_start))

                if x_end >= w:
                    break
                x += stride
            if y_end >= h:
                break
            y += stride

        return tiles

    def inspect_tiled(self, full_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Executes inference over overlapping tiles and maps coordinates to the full board.
        """
        h_full, w_full = full_image.shape[:2]
        total_board_area = float(h_full * w_full)

        # If image is smaller than tile size, run direct single-frame inspection
        if h_full <= self.tile_size and w_full <= self.tile_size:
            return self.inspector.inspect(full_image)

        tiles = self.slice_image(full_image)
        all_detections = []

        for tile, offset_x, offset_y in tiles:
            tile_dets = self.inspector.inspect(tile)
            for d in tile_dets:
                bbox = d.get("bbox", [])
                if len(bbox) == 4:
                    lx1, ly1, lx2, ly2 = bbox
                    # Translate to global coordinates
                    gx1 = float(lx1 + offset_x)
                    gy1 = float(ly1 + offset_y)
                    gx2 = float(lx2 + offset_x)
                    gy2 = float(ly2 + offset_y)

                    bw = gx2 - gx1
                    bh = gy2 - gy1
                    area_frac = (bw * bh) / total_board_area if total_board_area > 0 else 0.0

                    global_d = dict(d)
                    global_d["bbox"] = [gx1, gy1, gx2, gy2]
                    global_d["area_fraction"] = round(area_frac, 5)
                    all_detections.append(global_d)

        # Fused detections via global NMS
        fused = global_nms(all_detections, self.iou_threshold)
        return fused
