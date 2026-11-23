"""
YOLOv8 Post-Processing & Geometry Transformation for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Provides:
- Letterbox image padding & resizing
- Coordinate mapping back to original frame space
- Pure NumPy Non-Maximum Suppression (NMS)
- Output parsing for HailoRT tensors and ONNX tensors
- Deterministic defect class mapping
"""

import cv2
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np

CANONICAL_CLASSES = {
    0: "open_circuit",
    1: "short_circuit",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "pin_hole_missing_pad"
}

SEVERITY_BY_CLASS = {
    "open_circuit": "critical",
    "short_circuit": "critical",
    "pin_hole_missing_pad": "critical",
    "spur": "major",
    "mouse_bite": "major",
    "spurious_copper": "major"
}

# Verified HEF Quantization Parameters for per-tensor UINT8 dequantization fallback
HEF_QUANT_PARAMS = {
    # (stride, head_type): (scale, zero_point)
    (8, 'box'): (0.1211243868, 97.0),
    (8, 'cls'): (0.1861058474, 238.0),
    (16, 'box'): (0.1006753147, 93.0),
    (16, 'cls'): (0.1869599223, 242.0),
    (32, 'box'): (0.0808536783, 71.0),
    (32, 'cls'): (0.0744026601, 255.0)
}


def letterbox_image(
    image: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resizes image with aspect ratio preservation and pads with neutral color.
    Returns: (padded_image, scale_ratio, (pad_left, pad_top))
    """
    h, w = image.shape[:2]
    target_w, target_h = target_shape

    # Calculate scale factor
    r = min(target_w / w, target_h / h)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))

    # Compute padding
    dw = target_w - new_unpad_w
    dh = target_h - new_unpad_h
    pad_left = dw // 2
    pad_top = dh // 2
    pad_right = dw - pad_left
    pad_bottom = dh - pad_top

    if (w, h) != (new_unpad_w, new_unpad_h):
        resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=color
    )

    return padded, r, (pad_left, pad_top)


def unletterbox_boxes(
    boxes: np.ndarray,
    scale_ratio: float,
    pad: Tuple[int, int],
    orig_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Transforms [x1, y1, x2, y2] from letterboxed space back to original image space.
    Clips coordinates within original bounds.
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    pad_left, pad_top = pad
    orig_h, orig_w = orig_shape

    unletterboxed = boxes.copy()
    unletterboxed[:, [0, 2]] = (unletterboxed[:, [0, 2]] - pad_left) / scale_ratio
    unletterboxed[:, [1, 3]] = (unletterboxed[:, [1, 3]] - pad_top) / scale_ratio

    # Clip to valid pixel ranges
    unletterboxed[:, [0, 2]] = np.clip(unletterboxed[:, [0, 2]], 0, orig_w)
    unletterboxed[:, [1, 3]] = np.clip(unletterboxed[:, [1, 3]], 0, orig_h)

    return unletterboxed


def nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Pure NumPy implementation of Non-Maximum Suppression.
    boxes: np.ndarray of shape (N, 4) in [x1, y1, x2, y2]
    scores: np.ndarray of shape (N,)
    Returns: List of integer indices to keep.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def decode_yolov8_raw_heads(
    heads_dict: Dict[str, np.ndarray],
    reg_max: int = 16,
    num_classes: int = 6
) -> np.ndarray:
    """
    Decodes the 6 raw YOLOv8 convolution heads from HailoRT / DFC into the standard
    (1, 4 + num_classes, 8400) prediction tensor.
    Supports both NCHW and NHWC stream output formats and dynamically maps heads
    based on spatial dimension (stride 8: 80x80, stride 16: 40x40, stride 32: 20x20).
    """
    stride_map = {}
    for k, v in heads_dict.items():
        arr = np.array(v)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        # Detect channel position (NCHW vs NHWC)
        if arr.shape[0] in (64, num_classes):
            c, h, w = arr.shape
        elif arr.shape[-1] in (64, num_classes):
            arr = np.transpose(arr, (2, 0, 1))
            c, h, w = arr.shape
        else:
            continue

        stride = 640 // h
        if stride not in stride_map:
            stride_map[stride] = {}
        if c == 64:
            stride_map[stride]['box'] = arr
        elif c == num_classes:
            stride_map[stride]['cls'] = arr

    strides = [8, 16, 32]
    all_boxes = []
    all_scores = []
    proj = np.arange(reg_max, dtype=np.float32)

    for s in strides:
        if s not in stride_map or 'box' not in stride_map[s] or 'cls' not in stride_map[s]:
            raise ValueError(f"Missing required raw head stream for stride {s}. Available strides: {list(stride_map.keys())}")

        box_tensor = stride_map[s]['box']
        cls_tensor = stride_map[s]['cls']

        # If tensors are in UINT8, apply explicit per-tensor dequantization fallback
        if box_tensor.dtype == np.uint8:
            scale_b, zp_b = HEF_QUANT_PARAMS.get((s, 'box'), (1.0, 0.0))
            box_tensor = (box_tensor.astype(np.float32) - zp_b) * scale_b
        else:
            box_tensor = box_tensor.astype(np.float32, copy=False)

        if cls_tensor.dtype == np.uint8:
            scale_c, zp_c = HEF_QUANT_PARAMS.get((s, 'cls'), (1.0, 0.0))
            cls_tensor = (cls_tensor.astype(np.float32) - zp_c) * scale_c
        else:
            cls_tensor = cls_tensor.astype(np.float32, copy=False)

        h, w = box_tensor.shape[1], box_tensor.shape[2]

        # Numerically stable Sigmoid on class logits
        cls_tensor_clipped = np.clip(cls_tensor, -88.0, 88.0)
        cls_prob = 1.0 / (1.0 + np.exp(-cls_tensor_clipped))
        cls_prob = cls_prob.reshape(num_classes, -1)

        # Numerically stable DFL Softmax and expectation: (4, 16, H*W)
        box_reshaped = box_tensor.reshape(4, reg_max, -1)
        shifted_b = np.clip(box_reshaped - np.max(box_reshaped, axis=1, keepdims=True), -88.0, 0.0)
        exp_b = np.exp(shifted_b)
        sum_b = np.sum(exp_b, axis=1, keepdims=True)
        softmax_b = exp_b / np.maximum(sum_b, 1e-8)
        dist = np.tensordot(softmax_b, proj, axes=([1], [0]))

        xv, yv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        xv = xv.reshape(-1) + 0.5
        yv = yv.reshape(-1) + 0.5

        x1 = (xv - dist[0]) * s
        y1 = (yv - dist[1]) * s
        x2 = (xv + dist[2]) * s
        y2 = (yv + dist[3]) * s

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1

        all_boxes.append(np.stack([cx, cy, bw, bh], axis=0))
        all_scores.append(cls_prob)

    combined_boxes = np.concatenate(all_boxes, axis=1)
    combined_scores = np.concatenate(all_scores, axis=1)
    return np.expand_dims(np.concatenate([combined_boxes, combined_scores], axis=0), axis=0)


def decode_yolov8_output(
    output: Union[np.ndarray, Dict[str, np.ndarray]],
    orig_shape: Tuple[int, int],
    scale_ratio: float,
    pad: Tuple[int, int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_mapping: Optional[Dict[int, str]] = None
) -> List[Dict[str, Any]]:
    """
    Decodes YOLOv8 prediction tensor. Supports:
    1. Pre-decoded tensor: [1, 4 + num_classes, 8400] or [8400, 4 + num_classes] (Strategy A)
    2. Dict of 6 raw convolution heads from HailoRT / DFC (Strategy B)
    Returns standard EdgePCB-QC detection dictionaries.
    """
    classes = class_mapping or CANONICAL_CLASSES

    # If raw heads dictionary from Strategy B, decode to unified [1, 10, 8400] representation
    if isinstance(output, dict):
        output = decode_yolov8_raw_heads(output, num_classes=len(classes))

    # Squeeze batch dimension if present
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]

    # YOLOv8 standard output shape is (channels, anchors) -> transpose to (anchors, channels)
    if output.shape[0] < output.shape[1]:
        output = output.T

    # Coordinates: cx, cy, w, h
    cx = output[:, 0]
    cy = output[:, 1]
    w = output[:, 2]
    h = output[:, 3]

    # Class probabilities
    class_scores = output[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    # Confidence filter
    mask = max_scores >= conf_threshold
    if not np.any(mask):
        return []

    filtered_cx = cx[mask]
    filtered_cy = cy[mask]
    filtered_w = w[mask]
    filtered_h = h[mask]
    filtered_scores = max_scores[mask]
    filtered_class_ids = class_ids[mask]

    # Convert cx, cy, w, h to x1, y1, x2, y2 in letterbox space
    x1 = filtered_cx - (filtered_w / 2.0)
    y1 = filtered_cy - (filtered_h / 2.0)
    x2 = filtered_cx + (filtered_w / 2.0)
    y2 = filtered_cy + (filtered_h / 2.0)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Un-letterbox to original image coordinates
    unletterboxed_boxes = unletterbox_boxes(boxes, scale_ratio, pad, orig_shape)

    # Class-wise Non-Maximum Suppression
    final_detections = []
    unique_classes = np.unique(filtered_class_ids)

    for c in unique_classes:
        c_mask = filtered_class_ids == c
        c_boxes = unletterboxed_boxes[c_mask]
        c_scores = filtered_scores[c_mask]

        keep_indices = nms_numpy(c_boxes, c_scores, iou_threshold=iou_threshold)

        for idx in keep_indices:
            box = c_boxes[idx].tolist()
            conf = float(c_scores[idx])
            class_name = classes.get(int(c), f"class_{c}")
            severity = SEVERITY_BY_CLASS.get(class_name, "major")

            # Calculate area fraction relative to whole image
            img_h, img_w = orig_shape
            box_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
            area_frac = box_area / max(1.0, float(img_w * img_h))

            final_detections.append({
                "class_id": int(c),
                "class": class_name,
                "confidence": round(conf, 4),
                "bbox": [round(coord, 2) for coord in box],
                "severity": severity,
                "area_fraction": round(area_frac, 6)
            })

    # Sort descending by confidence
    final_detections.sort(key=lambda d: d["confidence"], reverse=True)
    return final_detections
