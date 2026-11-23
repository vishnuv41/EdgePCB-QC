"""
CPU Vision Inspector Backend for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Encapsulates the approved baseline PyTorch (.pt) / ONNX model running on the host CPU.
Provides:
- Synchronous inference on 640x640 or full-frame images
- Seamless fallback when Hailo hardware is not physically present
- Truthful execution latency measurement
- Dynamic threshold adjustment
"""

import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np

from inspection.cpu_inference import CPUInspector, CANONICAL_CLASSES

logger = logging.getLogger("edgepcb.inference.cpu")


class CpuBackend:
    """CPU inference backend wrapping the approved YOLOv8n baseline."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100
    ):
        self.inspector = CPUInspector(
            model_path=model_path,
            conf_threshold=conf_threshold
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def set_thresholds(
        self,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None
    ):
        """Updates inference thresholds dynamically."""
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)
            self.inspector.conf_threshold = float(conf_threshold)
        if iou_threshold is not None:
            self.iou_threshold = float(iou_threshold)
        if max_detections is not None:
            self.max_detections = int(max_detections)

    @property
    def model_path(self) -> Path:
        return self.inspector.model_path

    @property
    def is_available(self) -> bool:
        return self.inspector.is_loaded or self.inspector.yolo_model is not None or self.inspector.onnx_session is not None

    def inspect(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Executes CPU inference on a single image.
        Returns:
            dict containing detections, latency_ms, backend name, and model version.
        """
        t0 = time.perf_counter()
        detections = self.inspector.inspect(image)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Apply confidence filtering and max detections
        filtered = [d for d in detections if d.get("confidence", 0) >= self.conf_threshold]
        if len(filtered) > self.max_detections:
            filtered = filtered[:self.max_detections]

        return {
            "backend": "CPU (PyTorch / ONNX Baseline)",
            "model_version": "YOLOv8n-pcb-v1.0",
            "model_path": str(self.model_path),
            "model_name": Path(self.model_path).name if self.model_path else "unknown",
            "detections": filtered,
            "defect_count": len(filtered),
            "latency_ms": latency_ms,
            "hardware_accelerated": False
        }

    def get_info(self) -> Dict[str, Any]:
        """Returns backend metadata."""
        sha = "N/A"
        p = Path(self.model_path) if self.model_path else None
        if p and p.exists():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            sha = h.hexdigest()

        return {
            "backend": "cpu",
            "name": "CPUInspector (PyTorch / ONNX Runtime)",
            "model_path": str(self.model_path) if self.model_path else "None",
            "model_name": p.name if p else "None",
            "model_sha256": sha,
            "is_loaded": self.is_available,
            "confidence_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "hardware_accelerated": False
        }
