"""
Unified Inference Manager for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Coordinates vision inspection backends:
- auto: Automatically uses Hailo when hardware, runtime, and HEF are verified;
        otherwise cleanly falls back to CPU baseline with transparent diagnostic reasoning.
- cpu: Explicitly forces CPU-only inference (approved YOLOv8n baseline).
- hailo: Explicitly requests Hailo-8 NPU acceleration; reports warnings if hardware is missing.

Also provides:
- Live threshold tuning (confidence, NMS IoU, max detections)
- Runtime performance monitoring (rolling latency, FPS, defect counts)
- Side-by-side CPU vs. Hailo pipeline comparison & prediction agreement analytics
"""

import time
import logging
from collections import deque
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import numpy as np

from .cpu_backend import CpuBackend
from .hailo_backend import HailoBackend

logger = logging.getLogger("edgepcb.inference.manager")


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)

    return interArea / unionArea if unionArea > 0 else 0.0


class InferenceManager:
    """Manages model backend selection, hardware checking, and inspection routing."""

    DEFAULT_CONF = 0.25
    DEFAULT_IOU = 0.45
    DEFAULT_MAX_DETECTIONS = 100

    def __init__(
        self,
        backend_mode: str = "auto",
        conf_threshold: float = DEFAULT_CONF,
        iou_threshold: float = DEFAULT_IOU,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        cpu_model_path: Optional[Union[str, Path]] = None,
        hef_path: Optional[Union[str, Path]] = None
    ):
        self.backend_mode = backend_mode.lower()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        self.cpu_backend = CpuBackend(
            model_path=cpu_model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections
        )
        self.hailo_backend = HailoBackend(
            hef_path=hef_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections
        )

        self._active_backend_name = "cpu"
        self._fallback_reason = ""
        self._history = deque(maxlen=30)
        self._resolve_backend()

    def _resolve_backend(self):
        """Resolves which backend to activate based on configured mode and hardware reality."""
        mode = self.backend_mode.lower()

        if mode == "cpu":
            self._active_backend_name = "cpu"
            self._fallback_reason = "Manual CPU selection active."
        elif mode == "hailo":
            hw_avail = self.hailo_backend.is_hardware_available()
            hef_avail = self.hailo_backend.is_hef_available()
            if not hw_avail:
                self._fallback_reason = f"Hailo mode selected, but hardware unavailable ({self.hailo_backend._last_error})."
                logger.warning(self._fallback_reason)
            elif not hef_avail:
                self._fallback_reason = f"Hailo mode selected, but HEF model not found at {self.hailo_backend.hef_path}."
                logger.warning(self._fallback_reason)
            else:
                self._fallback_reason = ""
            self._active_backend_name = "hailo"
        else:  # auto
            hw_avail = self.hailo_backend.is_hardware_available()
            hef_avail = self.hailo_backend.is_hef_available()
            if hw_avail and hef_avail:
                self._active_backend_name = "hailo"
                self._fallback_reason = ""
                logger.info("Auto-selected Hailo NPU backend (hardware & HEF verified).")
            else:
                self._active_backend_name = "cpu"
                if not hw_avail:
                    self._fallback_reason = f"Hailo hardware not detected ({self.hailo_backend._last_error}); falling back to CPU."
                else:
                    self._fallback_reason = f"HEF model missing at {self.hailo_backend.hef_path}; falling back to CPU."
                logger.info(f"Auto-selected CPU backend: {self._fallback_reason}")

    def set_backend_mode(self, mode: str) -> Dict[str, Any]:
        """Changes backend mode ('auto', 'cpu', 'hailo') and reports state."""
        clean_mode = mode.lower().strip()
        if clean_mode not in ["auto", "cpu", "hailo"]:
            raise ValueError(f"Invalid mode '{mode}'. Supported modes: 'auto', 'cpu', 'hailo'")
        self.backend_mode = clean_mode
        self._resolve_backend()
        return self.get_status_info()

    def set_thresholds(
        self,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None
    ) -> Dict[str, Any]:
        """Dynamically tunes inference thresholds without retraining."""
        if conf_threshold is not None:
            self.conf_threshold = max(0.01, min(0.99, float(conf_threshold)))
        if iou_threshold is not None:
            self.iou_threshold = max(0.01, min(0.99, float(iou_threshold)))
        if max_detections is not None:
            self.max_detections = max(1, min(500, int(max_detections)))

        self.cpu_backend.set_thresholds(
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections
        )
        self.hailo_backend.set_thresholds(
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections
        )
        return self.get_config()

    def reset_thresholds(self) -> Dict[str, Any]:
        """Restores default inference thresholds."""
        return self.set_thresholds(
            conf_threshold=self.DEFAULT_CONF,
            iou_threshold=self.DEFAULT_IOU,
            max_detections=self.DEFAULT_MAX_DETECTIONS
        )

    def get_config(self) -> Dict[str, Any]:
        """Returns current configuration and threshold parameters."""
        return {
            "backend_mode": self.backend_mode,
            "active_backend": self._active_backend_name,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "default_conf": self.DEFAULT_CONF,
            "default_iou": self.DEFAULT_IOU,
            "default_max_detections": self.DEFAULT_MAX_DETECTIONS,
            "fallback_reason": self._fallback_reason
        }

    def inspect(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Executes PCB inspection using the resolved backend.
        Never falsely labels CPU inference as Hailo inference.
        """
        self._resolve_backend()
        t_start = time.perf_counter()

        if self._active_backend_name == "hailo":
            try:
                res = self.hailo_backend.inspect(image)
            except Exception as e:
                logger.error(f"Hailo inspection failed ({e}); falling back to CPU baseline.")
                res = self.cpu_backend.inspect(image)
                res["backend"] = "CPU (Fallback from Hailo error)"
                res["fallback_triggered"] = True
                res["fallback_reason"] = str(e)
        else:
            res = self.cpu_backend.inspect(image)

        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        res["total_pipeline_latency_ms"] = total_ms

        # Record metrics history
        self._history.append({
            "timestamp": time.time(),
            "inference_ms": res.get("latency_ms", 0),
            "total_ms": total_ms,
            "defects": res.get("defect_count", 0),
            "backend": res.get("backend", "unknown")
        })

        return res

    def compare_pipelines(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Runs the exact same image through both CPU and Hailo pipelines.
        Computes prediction agreement, IoU overlap, and latency differences.
        """
        # 1. Run CPU Pipeline
        cpu_t0 = time.perf_counter()
        cpu_res = self.cpu_backend.inspect(image)
        cpu_total_ms = round((time.perf_counter() - cpu_t0) * 1000, 2)

        # 2. Run or Probe Hailo Pipeline
        hailo_res = None
        hailo_error = None
        hailo_total_ms = 0.0

        if self.hailo_backend.is_hardware_available() and self.hailo_backend.is_hef_available():
            try:
                hailo_t0 = time.perf_counter()
                hailo_res = self.hailo_backend.inspect(image)
                hailo_total_ms = round((time.perf_counter() - hailo_t0) * 1000, 2)
            except Exception as e:
                hailo_error = str(e)
        else:
            hailo_error = self.hailo_backend._last_error or "Hailo-8 hardware or HEF not detected."

        # 3. Calculate Prediction Agreement
        cpu_dets = cpu_res.get("detections", [])
        hailo_dets = hailo_res.get("detections", []) if hailo_res else []

        matches = []
        ious = []
        unmatched_cpu = list(range(len(cpu_dets)))
        unmatched_hailo = list(range(len(hailo_dets)))

        for c_idx, c_det in enumerate(cpu_dets):
            c_box = c_det.get("bbox", [0, 0, 0, 0])
            c_cls = c_det.get("class", "")
            best_iou = 0.0
            best_h_idx = -1

            for h_idx in unmatched_hailo:
                h_det = hailo_dets[h_idx]
                h_box = h_det.get("bbox", [0, 0, 0, 0])
                iou = compute_iou(c_box, h_box)
                if iou > best_iou:
                    best_iou = iou
                    best_h_idx = h_idx

            if best_iou >= 0.40 and best_h_idx >= 0:
                h_det = hailo_dets[best_h_idx]
                class_matched = (c_cls == h_det.get("class", ""))
                matches.append({
                    "cpu_class": c_cls,
                    "hailo_class": h_det.get("class", ""),
                    "class_match": class_matched,
                    "iou": round(best_iou, 3),
                    "cpu_conf": round(c_det.get("confidence", 0), 3),
                    "hailo_conf": round(h_det.get("confidence", 0), 3)
                })
                ious.append(best_iou)
                unmatched_cpu.remove(c_idx)
                unmatched_hailo.remove(best_h_idx)

        mean_iou = round(float(np.mean(ious)), 3) if ious else 0.0
        class_agreement_pct = round(
            (sum(1 for m in matches if m["class_match"]) / max(1, len(matches))) * 100, 1
        ) if matches else (100.0 if len(cpu_dets) == 0 and len(hailo_dets) == 0 else 0.0)

        return {
            "sample_tested": str(image) if isinstance(image, (str, Path)) else "Raw Frame",
            "cpu_pipeline": {
                "available": True,
                "backend": cpu_res.get("backend", "CPU Baseline"),
                "model_name": cpu_res.get("model_name", "YOLOv8n"),
                "inference_latency_ms": cpu_res.get("latency_ms", 0),
                "total_latency_ms": cpu_total_ms,
                "estimated_fps": round(1000.0 / max(1.0, cpu_res.get("latency_ms", 1)), 1),
                "defect_count": len(cpu_dets),
                "detections": cpu_dets
            },
            "hailo_pipeline": {
                "available": hailo_res is not None,
                "backend": "Hailo-8 NPU (AI HAT+)" if hailo_res else "Unavailable",
                "model_name": self.hailo_backend.hef_path.name,
                "inference_latency_ms": hailo_res.get("latency_ms", 0) if hailo_res else None,
                "total_latency_ms": hailo_total_ms if hailo_res else None,
                "estimated_fps": round(1000.0 / max(1.0, hailo_res.get("latency_ms", 1)), 1) if hailo_res else 0.0,
                "defect_count": len(hailo_dets) if hailo_res else 0,
                "detections": hailo_dets,
                "error": hailo_error
            },
            "prediction_agreement": {
                "class_agreement_percent": class_agreement_pct,
                "mean_bounding_box_iou": mean_iou,
                "matched_count": len(matches),
                "cpu_only_count": len(unmatched_cpu),
                "hailo_only_count": len(unmatched_hailo),
                "matches": matches
            }
        }

    def get_status_info(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic status for dashboard and telemetry."""
        cpu_info = self.cpu_backend.get_info()
        hailo_info = self.hailo_backend.get_info()

        # Calculate rolling metrics
        if self._history:
            avg_inf = round(sum(h["inference_ms"] for h in self._history) / len(self._history), 1)
            avg_tot = round(sum(h["total_ms"] for h in self._history) / len(self._history), 1)
            avg_fps = round(1000.0 / max(1.0, avg_inf), 1)
        else:
            avg_inf = 0.0
            avg_tot = 0.0
            avg_fps = 0.0

        return {
            "configured_mode": self.backend_mode,
            "active_backend": self._active_backend_name,
            "active_backend_label": (
                "Hailo-8 / Hailo-8L NPU (AI HAT+)" if self._active_backend_name == "hailo"
                else "PyTorch / ONNX CPU Baseline"
            ),
            "hardware_accelerated": self._active_backend_name == "hailo",
            "fallback_reason": self._fallback_reason,
            "thresholds": {
                "confidence": self.conf_threshold,
                "iou": self.iou_threshold,
                "max_detections": self.max_detections
            },
            "performance_metrics": {
                "avg_inference_latency_ms": avg_inf,
                "avg_total_latency_ms": avg_tot,
                "avg_fps": avg_fps,
                "total_inspections_tracked": len(self._history)
            },
            "cpu_backend_status": cpu_info,
            "hailo_backend_status": hailo_info
        }
