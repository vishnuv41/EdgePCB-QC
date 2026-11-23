"""
USB / V4L2 Camera Interface for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Uses OpenCV VideoCapture over V4L2 (Video for Linux 2) on Raspberry Pi / Linux,
or DirectShow (CAP_DSHOW) on Windows workstations for stable development testing.
"""

import os
import cv2
import time
import logging
import threading
from typing import Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger("edgepcb.camera.usb")


class UsbCamera:
    """USB webcam / industrial UVC camera controller."""

    def __init__(
        self,
        device_index: int = 0,
        resolution: Tuple[int, int] = (1280, 720),
        framerate: int = 30
    ):
        self.device_index = device_index
        self.resolution = resolution
        self.framerate = framerate
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_connected = False
        self._last_error = ""
        self._actual_resolution = (0, 0)
        self._actual_fps = 0.0
        self._consecutive_drop_count = 0
        self._max_consecutive_drops = 10
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> bool:
        """Initializes and opens the USB camera cleanly."""
        with self._lock:
            if self._is_connected and self.cap is not None and self.cap.isOpened():
                return True

            try:
                # Select OS-appropriate backend directly
                if os.name == "nt":
                    # On Windows, DirectShow is fast, robust, and doesn't freeze camera graphs
                    self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(self.device_index)
                else:
                    # On Linux / Raspberry Pi, use V4L2
                    self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
                    if not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(self.device_index)

                if not self.cap.isOpened():
                    self._is_connected = False
                    self._last_error = f"Cannot open USB camera device index {self.device_index}."
                    logger.warning(self._last_error)
                    return False

                # Configure resolution and FPS
                # Prefer MJPG fourcc for Logitech and UVC webcams on Linux to optimize USB bus bandwidth
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception:
                    pass

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.framerate)

                # Query actual negotiated parameters
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(self.cap.get(cv2.CAP_PROP_FPS))
                self._actual_resolution = (w, h)
                self._actual_fps = fps if fps > 0 else float(self.framerate)

                # Warm-up / discard first 2 frames then check validity
                test_frame = None
                for _ in range(3):
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        break
                    time.sleep(0.05)

                if test_frame is None or test_frame.size == 0:
                    self._internal_disconnect()
                    self._last_error = f"USB camera {self.device_index} opened but failed to yield test frame."
                    logger.error(self._last_error)
                    return False

                self._is_connected = True
                self._consecutive_drop_count = 0
                self._last_error = ""
                logger.info(f"USB Camera {self.device_index} connected: {w}x{h} @ {fps}fps")
                return True
            except Exception as e:
                self._is_connected = False
                self._last_error = f"USB camera exception on index {self.device_index}: {e}"
                logger.error(self._last_error)
                self._internal_disconnect()
                return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Reads a frame from the USB camera with drop tolerance."""
        if not self._is_connected or self.cap is None:
            return None

        with self._lock:
            if not self.cap or not self.cap.isOpened():
                return None

            try:
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    self._consecutive_drop_count += 1
                    if self._consecutive_drop_count >= self._max_consecutive_drops:
                        self._is_connected = False
                        self._last_error = f"Camera stream lost after {self._consecutive_drop_count} consecutive missed frames."
                        logger.error(self._last_error)
                        self._internal_disconnect()
                    return None

                self._consecutive_drop_count = 0
                return frame
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"Error reading USB camera frame: {e}")
                return None

    def _internal_disconnect(self):
        """Releases the camera hardware without deadlock (assumes caller holds _lock)."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.debug(f"USB camera release error: {e}")
            self.cap = None
        self._is_connected = False

    def disconnect(self):
        """Public thread-safe release of the camera hardware."""
        with self._lock:
            self._internal_disconnect()

    def get_info(self) -> Dict[str, Any]:
        """Returns camera metadata."""
        return {
            "type": "USB / V4L2",
            "device_index": self.device_index,
            "connected": self._is_connected,
            "requested_resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "actual_resolution": f"{self._actual_resolution[0]}x{self._actual_resolution[1]}",
            "requested_fps": self.framerate,
            "actual_fps": self._actual_fps,
            "last_error": self._last_error
        }
