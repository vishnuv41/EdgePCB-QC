"""
HailoRT Hardware Vision Inspector Backend for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Integrates with HailoRT on target hardware:
- Raspberry Pi 5 + Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L)
- Uses compiled HEF (Hailo Executable Format) model.
- Strictly reports unavailable when physical Hailo hardware is absent.
- Thread-safe inference execution using a dedicated hardware mutex.
"""

import os
import time
import hashlib
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np

from .yolo_postprocess import letterbox_image, decode_yolov8_output, CANONICAL_CLASSES

logger = logging.getLogger("edgepcb.inference.hailo")

try:
    import hailo_platform
    HAS_HAILORT = True
    HAILORT_VERSION = getattr(hailo_platform, "__version__", "Available")
except ImportError:
    hailo_platform = None
    HAS_HAILORT = False
    HAILORT_VERSION = "Not Installed"

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
HAILO_DIR = WORKSPACE_ROOT / "models" / "hailo"
EXPECTED_HEF_NAME = "pcb_yolov8n_raw_heads.hef"
EXPECTED_HEF_SHA256 = "053a1967ad9489847ee80e058a4608518713706baf2fa79915159d952df3dada"


def discover_hef_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Discovers the active HEF path, prioritizing pcb_yolov8n_raw_heads.hef."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    # 1. Check expected primary HEF
    primary = HAILO_DIR / EXPECTED_HEF_NAME
    if primary.exists():
        return primary

    # 2. Check for other candidate HEFs
    candidates = list(HAILO_DIR.glob("*.hef"))
    if candidates:
        return sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]

    return primary


class HailoBackend:
    """Hailo-8 / Hailo-8L NPU inference backend using HailoRT."""

    def __init__(
        self,
        hef_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100
    ):
        self.hef_path = discover_hef_path(hef_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = None
        self.network_group = None
        self.hef = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        self._is_loaded = False
        self._lock = threading.Lock()
        self._last_error = ""

    def set_thresholds(self, conf_threshold: Optional[float] = None, iou_threshold: Optional[float] = None, max_detections: Optional[int] = None):
        """Updates inference thresholds dynamically."""
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)
        if iou_threshold is not None:
            self.iou_threshold = float(iou_threshold)
        if max_detections is not None:
            self.max_detections = int(max_detections)

    def is_hardware_available(self) -> bool:
        """Checks for HailoRT library and physical Hailo PCI accelerator device node."""
        # 1. Check for Linux device node /dev/hailo0
        dev_node = Path("/dev/hailo0")
        has_dev_node = dev_node.exists()

        if not HAS_HAILORT:
            if has_dev_node:
                self._last_error = "Hailo PCIe device node /dev/hailo0 exists, but HailoRT python package is missing."
            else:
                self._last_error = "HailoRT python package (hailo_platform) is not installed."
            return False

        try:
            devices = hailo_platform.Device.scan()
            if len(devices) == 0 and not has_dev_node:
                self._last_error = "HailoRT installed, but no Hailo PCIe accelerator device detected (/dev/hailo0 absent)."
                return False
            return True
        except Exception as e:
            if has_dev_node:
                return True
            self._last_error = f"Error scanning for Hailo devices: {e}"
            return False

    def is_hef_available(self) -> bool:
        """Verifies whether the compiled HEF model exists on disk."""
        return self.hef_path.exists() and self.hef_path.stat().st_size > 1024

    def get_hef_checksum(self) -> str:
        """Calculates SHA-256 checksum of the current HEF."""
        if not self.is_hef_available():
            return "MISSING"
        h = hashlib.sha256()
        with open(self.hef_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def is_checksum_valid(self) -> bool:
        """Compares current HEF checksum against expected target."""
        actual = self.get_hef_checksum()
        return actual.lower() == EXPECTED_HEF_SHA256.lower()

    def load_model(self) -> bool:
        """
        Configures the Hailo target device and loads the HEF file into NPU memory.
        """
        with self._lock:
            if not self.is_hardware_available():
                logger.warning(f"Cannot load Hailo model: {self._last_error}")
                return False

            if not self.is_hef_available():
                self._last_error = f"HEF file not found at {self.hef_path}. Model compilation required."
                logger.warning(self._last_error)
                return False

            try:
                # HailoRT 4.24: configure HEF through VDevice
                self.device = hailo_platform.VDevice()
                self.hef = hailo_platform.HEF(str(self.hef_path.resolve()))

                configure_params = hailo_platform.ConfigureParams.create_from_hef(
                    hef=self.hef,
                    interface=hailo_platform.HailoStreamInterface.PCIe
                )
                self.network_group = self.device.configure(
                    self.hef, configure_params
                )[0]

                self.input_vstreams_params = hailo_platform.InputVStreamParams.make(self.network_group)
                self.output_vstreams_params = hailo_platform.OutputVStreamParams.make(
                    self.network_group,
                    format_type=hailo_platform.FormatType.FLOAT32
                )

                self._is_loaded = True
                self._last_error = ""
                logger.info(f"Hailo HEF model loaded successfully: {self.hef_path.name}")
                return True
            except Exception as e:
                self._is_loaded = False
                self._last_error = f"Failed to initialize Hailo NPU runtime: {e}"
                logger.error(self._last_error)
                return False

    def close(self):
        """Releases Hailo VDevice and loaded HEF network group."""
        with self._lock:
            self.network_group = None
            self.input_vstreams_params = None
            self.output_vstreams_params = None
            self.hef = None
            if self.device is not None:
                try:
                    del self.device
                except Exception:
                    pass
                self.device = None
            self._is_loaded = False

    def inspect(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Executes NPU inference on Hailo AI HAT+ with thread-safe VStream synchronization.
        """
        if not self._is_loaded:
            if not self.load_model():
                raise RuntimeError(
                    f"Hailo backend unavailable: {self._last_error}. "
                    "Ensure Raspberry Pi AI HAT+ is seated and HEF model is compiled."
                )

        import cv2
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            img = image.copy()

        orig_shape = img.shape[:2]

        # Preprocessing: letterbox to 640x640 with neutral padding (114, 114, 114)
        letterboxed, scale, pad = letterbox_image(img, target_shape=(640, 640))
        # Convert BGR to RGB for YOLOv8 model input format (UINT8, NHWC)
        rgb_letterboxed = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb_letterboxed, axis=0)

        # Thread-safe NPU inference
        with self._lock:
            t0 = time.perf_counter()
            with self.network_group.activate(self.network_group.create_params()):
                with hailo_platform.InferVStreams(
                    self.network_group,
                    self.input_vstreams_params,
                    self.output_vstreams_params
                ) as infer_pipeline:
                    input_name = list(self.input_vstreams_params.keys())[0]
                    infer_results = infer_pipeline.infer({input_name: input_data})
                    if len(infer_results) == 1:
                        output_name = list(self.output_vstreams_params.keys())[0]
                        raw_output = infer_results[output_name]
                    else:
                        raw_output = infer_results
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Post-processing
        detections = decode_yolov8_output(
            raw_output,
            orig_shape=orig_shape,
            scale_ratio=scale,
            pad=pad,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )
        if len(detections) > self.max_detections:
            detections = detections[:self.max_detections]

        return {
            "backend": "Hailo-8 / Hailo-8L NPU (AI HAT+)",
            "model_version": "YOLOv8n-pcb-v1.0 (HEF)",
            "model_path": str(self.hef_path),
            "detections": detections,
            "defect_count": len(detections),
            "latency_ms": latency_ms,
            "hardware_accelerated": True
        }

    def get_info(self) -> Dict[str, Any]:
        """Returns Hailo backend status and diagnostic info."""
        hw_avail = self.is_hardware_available()
        hef_avail = self.is_hef_available()
        checksum = self.get_hef_checksum()
        checksum_valid = self.is_checksum_valid()

        return {
            "backend": "hailo",
            "name": "HailoInspector (Hailo-8 / Hailo-8L NPU)",
            "model_version": "YOLOv8n-pcb-v1.0 (HEF)",
            "hef_name": self.hef_path.name,
            "hef_path": str(self.hef_path),
            "expected_hef_name": EXPECTED_HEF_NAME,
            "expected_hef_sha256": EXPECTED_HEF_SHA256,
            "actual_hef_sha256": checksum,
            "checksum_valid": checksum_valid,
            "hailort_installed": HAS_HAILORT,
            "hailort_version": HAILORT_VERSION,
            "hardware_detected": hw_avail,
            "hef_file_found": hef_avail,
            "is_loaded": self._is_loaded,
            "confidence_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "hardware_accelerated": True,
            "last_error": self._last_error
        }
