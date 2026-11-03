"""
Vision Backend Abstraction Interface for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Provides common interface for:
- CPUInspector: Laptop development (PyTorch / ONNX / OpenCV)
- HailoInspector: Target hardware (Raspberry Pi 5 + Hailo-8/8L AI HAT+)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np


class VisionInspector(ABC):
    """Abstract Base Class for EdgePCB-QC Vision Backends."""

    def __init__(self, model_path: str = None, conf_threshold: float = 0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._last_results: List[Dict[str, Any]] = []
        self._last_timing_ms: float = 0.0

    @abstractmethod
    def load_model(self):
        """Loads and initializes the neural model weights or compiled graph."""
        pass

    @abstractmethod
    def inspect(self, image: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Runs visual inspection on a PCB image.
        Returns list of standardized defect dicts:
        [
            {
                "class": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2],
                "area_fraction": float,
                "location": str
            }
        ]
        """
        pass

    @abstractmethod
    def is_hardware_available(self) -> bool:
        """Returns True if the backend hardware (e.g. Hailo NPU) is present."""
        pass

    def get_last_results(self) -> List[Dict[str, Any]]:
        return self._last_results

    def get_last_timing(self) -> float:
        return self._last_timing_ms
