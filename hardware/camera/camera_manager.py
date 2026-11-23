"""
Persistent Camera Manager & Hardware Abstraction for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Architectural guarantees:
- Process-level singleton (persists across Streamlit reruns and tab navigation)
- Safe discovery: NEVER probes or releases currently active streaming camera devices
- DirectShow (Windows) and V4L2 (Linux) backend selection
- Bounded single-frame buffer (zero frame backlog, zero latency buildup, zero memory leak)
- Automatic shutdown registration via atexit
- Strictly honest status reporting (never fabricates video frames)
"""

import os
import cv2
import time
import glob
import atexit
import logging
import threading
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .csi_camera import CsiCamera, HAS_PICAMERA2
from .usb_camera import UsbCamera

logger = logging.getLogger("edgepcb.camera.manager")


class CameraStatus(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    STREAMING = "STREAMING"
    ERROR = "ERROR"


class MockCamera:
    """Synthetic Diagnostic Simulator for offline CI testing and test pattern validation."""

    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 640),
        framerate: int = 15,
        sample_image_dir: Optional[Path] = None
    ):
        self.resolution = resolution
        self.framerate = framerate
        self._is_connected = False
        self._sample_dir = sample_image_dir
        self._sample_files: List[Path] = []
        self._sample_idx = 0

        if self._sample_dir and self._sample_dir.exists():
            self._sample_files = list(self._sample_dir.glob("*.jpg")) + list(self._sample_dir.glob("*.png"))

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> bool:
        self._is_connected = True
        return True

    def get_frame(self) -> Optional[np.ndarray]:
        if not self._is_connected:
            return None

        if self._sample_files:
            file_path = self._sample_files[self._sample_idx % len(self._sample_files)]
            img = cv2.imread(str(file_path))
            if img is not None:
                if (img.shape[1], img.shape[0]) != self.resolution:
                    img = cv2.resize(img, self.resolution)
                return img

        # Generate a distinct synthetic diagnostic test pattern with PCB traces and moving timestamp
        canvas = np.full((self.resolution[1], self.resolution[0], 3), (25, 90, 35), dtype=np.uint8)
        cv2.line(canvas, (50, 150), (self.resolution[0] - 50, 150), (0, 215, 255), 4)
        cv2.line(canvas, (50, 300), (self.resolution[0] - 50, 300), (0, 215, 255), 4)
        cv2.circle(canvas, (200, 225), 18, (0, 215, 255), -1)
        cv2.circle(canvas, (200, 225), 8, (15, 45, 20), -1)

        ts_str = time.strftime("%H:%M:%S", time.localtime()) + f".{int((time.time() % 1) * 100):02d}"
        cv2.putText(canvas, f"SYNTHETIC DIAGNOSTIC PATTERN {ts_str}", (20, self.resolution[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas

    def disconnect(self):
        self._is_connected = False

    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "SYNTHETIC_SIMULATOR (Offline Diagnostic Pattern)",
            "connected": self._is_connected,
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "requested_fps": self.framerate,
            "last_error": ""
        }


class CameraManager:
    """
    Central persistent camera controller managing lifecycle, background thread,
    and bounded latest-frame buffer.
    """

    def __init__(self):
        self._active_camera: Optional[Any] = None
        self._camera_id: str = "none"
        self._camera_type: str = "none"
        self._status: CameraStatus = CameraStatus.DISCONNECTED
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._target_fps: int = 15
        self._frame_count: int = 0
        self._actual_fps: float = 0.0
        self._last_fps_time: float = time.time()
        self._last_error: str = ""
        self._cached_cameras: Optional[List[Dict[str, Any]]] = None
        self._last_scan_time: float = 0.0
        self._current_resolution: Tuple[int, int] = (1280, 720)

    @property
    def status(self) -> CameraStatus:
        return self._status

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def camera_type(self) -> str:
        return self._camera_type

    @property
    def is_streaming(self) -> bool:
        return self._status == CameraStatus.STREAMING and self._active_camera is not None and self._active_camera.is_connected

    def _find_working_usb_camera(self, resolution: Tuple[int, int], framerate: int, preferred_index: Optional[int] = None) -> Optional[UsbCamera]:
        """
        Scans for an operational USB/Logitech webcam that can successfully connect
        and yield a real video frame. Crucial on Linux where USB webcams may appear
        at /dev/video0, /dev/video2, /dev/video4, etc.
        """
        candidates = []
        if preferred_index is not None:
            candidates.append(preferred_index)

        # Environment variable override (e.g. EDGEPCB_CAMERA_INDEX=2)
        env_idx = os.environ.get("EDGEPCB_CAMERA_INDEX")
        if env_idx is not None and env_idx.isdigit():
            idx = int(env_idx)
            if idx not in candidates:
                candidates.append(idx)

        # Linux V4L2 device discovery
        if os.name != "nt":
            video_nodes = sorted(glob.glob("/dev/video*"), key=lambda x: int(x.replace("/dev/video", "")) if x.replace("/dev/video", "").isdigit() else 999)
            usb_indices = []
            other_indices = []
            for node in video_nodes:
                try:
                    idx = int(node.replace("/dev/video", ""))
                    name_p = Path(f"/sys/class/video4linux/{Path(node).name}/name")
                    dev_name = name_p.read_text(encoding="utf-8").lower() if name_p.exists() else ""
                    # Ignore internal Raspberry Pi hardware codecs/ISP/CFE
                    if any(x in dev_name for x in ["codec", "m2m", "isp", "unicam-image", "pisp", "rp1-cfe", "cfe", "hevc"]):
                        continue
                    if any(x in dev_name for x in ["brio", "logitech", "webcam", "uvc", "camera"]):
                        usb_indices.append(idx)
                    else:
                        other_indices.append(idx)
                except Exception:
                    pass
            for idx in usb_indices + other_indices:
                if idx not in candidates:
                    candidates.append(idx)
        else:
            for idx in range(4):
                if idx not in candidates:
                    candidates.append(idx)

        # General fallbacks
        for idx in range(8):
            if idx not in candidates:
                candidates.append(idx)

        for idx in candidates:
            try:
                cam = UsbCamera(device_index=idx, resolution=resolution, framerate=framerate)
                if cam.connect():
                    logger.info(f"[+] Auto-detected and verified working USB camera at index {idx}")
                    return cam
            except Exception as e:
                logger.debug(f"Candidate camera index {idx} failed: {e}")

        return None

    def discover_cameras(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Discovers physically accessible cameras.
        Crucial:
        - Never probes or reopens an index that is currently active and streaming.
        - Caches results persistently until force_refresh=True is explicitly requested,
          preventing unwanted camera hardware activation or privacy LED blinking on Streamlit reruns.
        - Does NOT call cap.read() during enumeration to avoid powering up the camera sensor.
        """
        now = time.time()
        if not force_refresh and self._cached_cameras is not None:
            # If camera is streaming, ensure the active camera status is reflected accurately
            for c in self._cached_cameras:
                if c["id"] == self._camera_id:
                    c["available"] = self.is_streaming
            return self._cached_cameras

        discovered = []

        # 1. Raspberry Pi CSI Camera
        if HAS_PICAMERA2:
            try:
                from .csi_camera import find_csi_camera_index
                csi_idx = find_csi_camera_index()
                is_streaming_this = (self.is_streaming and self._camera_type == "csi")
                status_suffix = " [Streaming]" if is_streaming_this else ""
                
                discovered.append({
                    "id": f"csi:{csi_idx}",
                    "name": f"Raspberry Pi CSI Camera (OV5647 Rev 1.3){status_suffix}",
                    "type": "csi",
                    "device_index": csi_idx,
                    "available": True,
                    "recommended_for": "Production Raspberry Pi 5 Inspection (CSI-MIPI)"
                })
            except Exception as e:
                discovered.append({
                    "id": "csi:0",
                    "name": "Raspberry Pi CSI Camera (Uninitialized)",
                    "type": "csi",
                    "device_index": 0,
                    "available": False,
                    "error": str(e)
                })

        # 2. USB / V4L2 / DirectShow Cameras
        if os.name != "nt":
            video_nodes = sorted(glob.glob("/dev/video*"), key=lambda x: int(x.replace("/dev/video", "")) if x.replace("/dev/video", "").isdigit() else 999)
            for node in video_nodes:
                try:
                    idx = int(node.replace("/dev/video", ""))
                    name_p = Path(f"/sys/class/video4linux/{Path(node).name}/name")
                    raw_name = name_p.read_text(encoding="utf-8").strip() if name_p.exists() else f"Video Device ({node})"

                    # Ignore internal hardware codecs / ISP / CFE on Raspberry Pi
                    if any(x in raw_name.lower() for x in ["codec", "m2m", "isp", "unicam-image", "pisp", "rp1-cfe", "cfe", "hevc"]):
                        continue

                    is_brio = "brio" in raw_name.lower()
                    is_logitech = "logitech" in raw_name.lower() or is_brio
                    if is_brio:
                        friendly_name = f"Logitech Brio 100 Full HD Webcam ({node})"
                    else:
                        friendly_name = f"{raw_name} ({node})"
                    rec_for = "Logitech USB Inspection Webcam" if is_logitech else "Benchtop USB Inspection / Webcam"

                    if self.is_streaming and self._camera_type == "usb" and getattr(self._active_camera, "device_index", -1) == idx:
                        discovered.append({
                            "id": f"usb:{idx}",
                            "name": f"{friendly_name} [Streaming]",
                            "type": "usb",
                            "device_index": idx,
                            "available": True,
                            "recommended_for": rec_for
                        })
                        continue

                    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                    if cap.isOpened():
                        discovered.append({
                            "id": f"usb:{idx}",
                            "name": friendly_name,
                            "type": "usb",
                            "device_index": idx,
                            "available": True,
                            "recommended_for": rec_for
                        })
                    cap.release()
                except Exception:
                    pass
        else:
            # Windows workstation probe (DirectShow)
            for idx in range(3):
                # If this index is currently in use by active streaming camera, DO NOT touch it!
                if self.is_streaming and self._camera_type == "usb" and getattr(self._active_camera, "device_index", -1) == idx:
                    discovered.append({
                        "id": f"usb:{idx}",
                        "name": f"Laptop / USB Webcam (Index {idx}) [Streaming]",
                        "type": "usb",
                        "device_index": idx,
                        "available": True,
                        "recommended_for": "Development Webcam / USB Camera"
                    })
                    continue

                try:
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        cam_name = "Laptop Built-in Webcam" if idx == 0 else f"USB Video Device (Index {idx})"
                        discovered.append({
                            "id": f"usb:{idx}",
                            "name": f"{cam_name} (Index {idx})",
                            "type": "usb",
                            "device_index": idx,
                            "available": True,
                            "recommended_for": "Development Webcam / USB Camera"
                        })
                    cap.release()
                except Exception:
                    pass

        # 3. Add Diagnostic Synthetic Pattern option (always listed last)
        discovered.append({
            "id": "mock:synthetic",
            "name": "Synthetic Diagnostic Simulator (Offline CI Test Pattern)",
            "type": "mock",
            "device_index": -1,
            "available": True,
            "recommended_for": "Offline testing without physical camera"
        })

        self._cached_cameras = discovered
        self._last_scan_time = now
        return discovered

    def start_camera(
        self,
        camera_id: str = "usb:0",
        resolution: Tuple[int, int] = (1280, 720),
        framerate: int = 15
    ) -> bool:
        """
        Starts camera stream with idempotency check.
        If camera is already running on this device and resolution, does nothing and returns True.
        """
        # Idempotency: if already streaming on the same camera and settings, do not disrupt!
        if (self.is_streaming and
            self._camera_id == camera_id and
            self._current_resolution == resolution and
            self._target_fps == framerate):
            logger.debug(f"Camera {camera_id} is already streaming cleanly; keeping active.")
            return True

        # Stop existing camera cleanly if switching device
        self.stop_camera()

        self._status = CameraStatus.CONNECTING
        self._target_fps = framerate
        self._current_resolution = resolution

        # Resolve device with intelligent Logitech / USB auto-detection & fallback
        cam_str = str(camera_id).lower().strip() if camera_id is not None else "auto"

        if cam_str == "mock:synthetic":
            self._active_camera = MockCamera(resolution=resolution, framerate=framerate)
            self._camera_type = "mock"
            self._camera_id = "mock:synthetic"
            if not self._active_camera.connect():
                self._status = CameraStatus.ERROR
                self._last_error = "Failed to connect to synthetic simulator."
                self._active_camera = None
                return False

        elif cam_str.startswith("csi:"):
            idx = int(cam_str.split(":")[1])
            self._active_camera = CsiCamera(camera_num=idx, resolution=resolution, framerate=framerate)
            self._camera_type = "csi"
            self._camera_id = f"csi:{idx}"
            if not self._active_camera.connect():
                logger.warning(f"CSI camera {idx} failed. Attempting USB camera auto-fallback...")
                fallback_usb = self._find_working_usb_camera(resolution, framerate)
                if fallback_usb:
                    self._active_camera = fallback_usb
                    self._camera_type = "usb"
                    self._camera_id = f"usb:{fallback_usb.device_index}"
                else:
                    self._status = CameraStatus.ERROR
                    self._last_error = self._active_camera.get_info().get("last_error", "CSI camera failed.")
                    self._active_camera = None
                    return False

        elif cam_str.startswith("usb:") or cam_str.isdigit():
            idx = int(cam_str.split(":")[1]) if ":" in cam_str else int(cam_str)
            cam = UsbCamera(device_index=idx, resolution=resolution, framerate=framerate)
            if cam.connect():
                self._active_camera = cam
                self._camera_type = "usb"
                self._camera_id = f"usb:{idx}"
            else:
                logger.warning(f"Requested USB camera index {idx} failed. Auto-scanning for operational USB/Logitech webcam...")
                fallback_usb = self._find_working_usb_camera(resolution, framerate, preferred_index=idx)
                if fallback_usb:
                    self._active_camera = fallback_usb
                    self._camera_type = "usb"
                    self._camera_id = f"usb:{fallback_usb.device_index}"
                else:
                    self._status = CameraStatus.ERROR
                    self._last_error = f"Cannot open USB camera (tried index {idx} and all available /dev/video* devices)."
                    self._active_camera = None
                    return False

        elif cam_str in ("auto", "none", "", "default"):
            # Auto-detect camera: prioritize physical USB / Logitech webcams, then CSI
            detected_cam = self._find_working_usb_camera(resolution, framerate)
            if detected_cam:
                self._active_camera = detected_cam
                self._camera_type = "usb"
                self._camera_id = f"usb:{detected_cam.device_index}"
            elif HAS_PICAMERA2:
                try:
                    csi = CsiCamera(camera_num=0, resolution=resolution, framerate=framerate)
                    if csi.connect():
                        self._active_camera = csi
                        self._camera_type = "csi"
                        self._camera_id = "csi:0"
                except Exception:
                    pass

            if not self._active_camera:
                self._status = CameraStatus.ERROR
                self._last_error = "No operational camera detected (USB/Logitech or CSI). Please check connection and permissions."
                self._active_camera = None
                return False
        else:
            self._status = CameraStatus.ERROR
            self._last_error = f"Unrecognized camera identifier: '{camera_id}'"
            return False

        # Launch background frame-capture thread
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_worker,
            daemon=True,
            name="CameraCaptureWorker"
        )
        self._capture_thread.start()

        self._status = CameraStatus.STREAMING
        self._last_error = ""
        # Refresh cache to show active streaming
        self._cached_cameras = None
        logger.info(f"Camera streaming started successfully on {camera_id}")
        return True

    def _capture_worker(self):
        """Worker thread continuously polling camera into bounded latest-frame buffer."""
        fps_interval = 1.0 / max(1, self._target_fps)
        frames_in_interval = 0
        last_calc = time.time()
        consecutive_failures = 0

        while not self._stop_event.is_set() and self._active_camera and self._active_camera.is_connected:
            t0 = time.time()
            frame = self._active_camera.get_frame()

            if frame is not None and frame.size > 0:
                with self._lock:
                    self._latest_frame = frame
                frames_in_interval += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 15:
                    logger.warning("Camera worker reached 15 consecutive missed frames; stopping stream.")
                    self._status = CameraStatus.ERROR
                    self._last_error = "Camera stream lost (device disconnected or unavailable)."
                    break
                time.sleep(0.05)

            # Update measured FPS once per second
            now = time.time()
            if now - last_calc >= 1.0:
                self._actual_fps = frames_in_interval / (now - last_calc)
                frames_in_interval = 0
                last_calc = now

            elapsed = time.time() - t0
            sleep_time = max(0.001, fps_interval - elapsed)
            time.sleep(sleep_time)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Returns a copy of the latest captured frame.
        Guaranteed non-blocking, zero frame backlog, zero lag.
        """
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def capture_still(self) -> Optional[np.ndarray]:
        """
        Captures a high-fidelity still frame for formal inspection.
        Grabs latest frame non-destructively so the live stream is never disrupted.
        """
        frame = self.get_latest_frame()
        if frame is not None:
            return frame
        if self._active_camera and self._active_camera.is_connected:
            return self._active_camera.get_frame()
        return None

    def stop_camera(self):
        """Safely stops streaming thread and releases camera hardware."""
        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        self._capture_thread = None

        if self._active_camera:
            try:
                self._active_camera.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting camera: {e}")
            self._active_camera = None

        with self._lock:
            self._latest_frame = None

        self._status = CameraStatus.DISCONNECTED
        self._camera_id = "none"
        self._camera_type = "none"
        self._actual_fps = 0.0
        self._cached_cameras = None

    def get_image_quality_metrics(self) -> Dict[str, Any]:
        """Calculates optical sharpness, brightness, and contrast from the latest live frame."""
        with self._lock:
            if self._latest_frame is None:
                return {
                    "status": "standby",
                    "sharpness_laplacian": 0.0,
                    "mean_brightness": 0.0,
                    "contrast_std": 0.0,
                    "focus_quality": "STANDBY"
                }
            frame = self._latest_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        lap_var = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
        mean_b = round(float(np.mean(gray)), 2)
        contrast = round(float(np.std(gray)), 2)

        if lap_var >= 60.0:
            focus = "GOOD"
        elif lap_var >= 25.0:
            focus = "ACCEPTABLE"
        else:
            focus = "BLURRY"

        return {
            "status": "active",
            "sharpness_laplacian": lap_var,
            "mean_brightness": mean_b,
            "contrast_std": contrast,
            "focus_quality": focus
        }

    def get_status_info(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic status for dashboard telemetry."""
        info = {
            "status": self._status.value,
            "camera_id": self._camera_id,
            "camera_type": self._camera_type,
            "target_fps": self._target_fps,
            "measured_fps": round(self._actual_fps, 1),
            "is_streaming": self.is_streaming,
            "resolution": f"{self._current_resolution[0]}x{self._current_resolution[1]}",
            "last_error": self._last_error,
            "quality_metrics": self.get_image_quality_metrics()
        }
        if self._active_camera:
            info.update(self._active_camera.get_info())
        return info


# Process-level singleton instance
_GLOBAL_CAMERA_MANAGER: Optional[CameraManager] = None
_SINGLETON_LOCK = threading.Lock()


def get_camera_manager() -> CameraManager:
    """Returns the persistent process singleton CameraManager."""
    global _GLOBAL_CAMERA_MANAGER
    with _SINGLETON_LOCK:
        if _GLOBAL_CAMERA_MANAGER is None:
            _GLOBAL_CAMERA_MANAGER = CameraManager()
            atexit.register(_cleanup_global_camera)
        return _GLOBAL_CAMERA_MANAGER


def _cleanup_global_camera():
    global _GLOBAL_CAMERA_MANAGER
    if _GLOBAL_CAMERA_MANAGER is not None:
        try:
            _GLOBAL_CAMERA_MANAGER.stop_camera()
        except Exception:
            pass
