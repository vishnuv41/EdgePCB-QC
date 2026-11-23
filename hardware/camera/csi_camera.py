"""
CSI Camera Interface for Raspberry Pi Camera Module 3
Bharat AI-SoC Challenge 2026-27 | PS5

Uses picamera2 (official libcamera Python binding for Raspberry Pi OS 64-bit).
Supports Sony IMX708 sensor features:
- Hardware autofocus (AF)
- Full HD / 2K preview and full-res capture
- RGB888 / BGR888 frame capture
"""

import time
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger("edgepcb.camera.csi")

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    Picamera2 = None
    HAS_PICAMERA2 = False


def find_csi_camera_index() -> int:
    """
    Finds the index of the CSI camera in libcamera.
    On Pi 5 with a USB webcam attached, Camera 0 may be UVC and Camera 1 may be CSI.
    """
    try:
        import libcamera
        cm = libcamera.CameraManager.singleton()
        for idx, cam in enumerate(cm.cameras):
            cid = cam.id.lower()
            if any(s in cid for s in ["ov5647", "imx708", "imx219", "imx477", "i2c", "rpi"]):
                return idx
        if len(cm.cameras) > 1:
            return 1
    except Exception:
        pass
    return 0


class CsiCamera:
    """Raspberry Pi CSI Camera controller (supports OV5647 Rev 1.3 and IMX708 Module 3) using Picamera2."""

    def __init__(
        self,
        camera_num: Optional[int] = None,
        resolution: Tuple[int, int] = (1920, 1080),
        framerate: int = 15,
        enable_autofocus: bool = False
    ):
        if camera_num is None:
            self.camera_num = find_csi_camera_index()
        else:
            self.camera_num = camera_num
        self.resolution = resolution
        self.framerate = framerate
        self.enable_autofocus = enable_autofocus
        self.camera: Optional[Any] = None
        self._is_connected = False
        self._sensor_name = "Raspberry Pi CSI Camera (OV5647 Rev 1.3)"
        self._last_error = ""

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> bool:
        """Initializes the Picamera2 device on Raspberry Pi 5."""
        if not HAS_PICAMERA2:
            self._last_error = (
                "picamera2 library not found. Picamera2 is pre-installed on "
                "Raspberry Pi OS (64-bit Bookworm). On non-Pi systems, CSI camera is unavailable."
            )
            logger.warning(self._last_error)
            self._is_connected = False
            return False

        try:
            self.camera = Picamera2(self.camera_num)
            try:
                config = self.camera.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"},
                    controls={"FrameRate": self.framerate}
                )
                self.camera.configure(config)
            except Exception:
                # Sensors like OV5647 on Pi 5 may not support FrameDurationLimits
                config = self.camera.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"}
                )
                self.camera.configure(config)

            if self.enable_autofocus:
                try:
                    # Camera Module 3 IMX708 AF mode continuous
                    from picamera2.controls import AfModeEnum
                    self.camera.set_controls({"AfMode": AfModeEnum.Continuous})
                except Exception as af_err:
                    logger.debug(f"Autofocus control configuration note: {af_err}")

            self.camera.start()
            self._is_connected = True
            self._last_error = ""
            logger.info(f"CSI Camera {self.camera_num} connected successfully at {self.resolution} @ {self.framerate}fps")
            return True
        except Exception as e:
            self._is_connected = False
            self._last_error = f"Failed to open CSI Camera {self.camera_num}: {e}"
            logger.error(self._last_error)
            if self.camera:
                try:
                    self.camera.stop()
                    self.camera.close()
                except Exception:
                    pass
                self.camera = None
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Captures a single BGR frame from the CSI camera."""
        if not self._is_connected or self.camera is None:
            return None

        try:
            # Picamera2 capture_array returns RGB888 numpy array
            rgb_frame = self.camera.capture_array("main")
            if rgb_frame is None or rgb_frame.size == 0:
                return None
            # Convert RGB to BGR for OpenCV compatibility
            bgr_frame = rgb_frame[..., ::-1].copy()
            return bgr_frame
        except Exception as e:
            logger.error(f"Error reading frame from CSI camera: {e}")
            self._last_error = str(e)
            return None

    def trigger_autofocus(self):
        """Triggers manual one-shot autofocus cycle if supported."""
        if self._is_connected and self.camera:
            try:
                from picamera2.controls import AfTriggerEnum
                self.camera.set_controls({"AfTrigger": AfTriggerEnum.Start})
            except Exception as e:
                logger.debug(f"AF trigger not available: {e}")

    def disconnect(self):
        """Stops and releases the CSI camera."""
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception as e:
                logger.debug(f"CSI camera close error: {e}")
            self.camera = None
        self._is_connected = False

    def get_info(self) -> Dict[str, Any]:
        """Returns camera metadata."""
        return {
            "type": f"CSI ({self._sensor_name})",
            "camera_num": self.camera_num,
            "connected": self._is_connected,
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "target_fps": self.framerate,
            "has_picamera2": HAS_PICAMERA2,
            "autofocus_enabled": self.enable_autofocus,
            "device_name": self._sensor_name,
            "last_error": self._last_error
        }
