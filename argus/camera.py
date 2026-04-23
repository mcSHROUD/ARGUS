from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import cv2
import numpy as np

from argus.config import BurstConfig, CameraConfig, MotionConfig

logger = logging.getLogger(__name__)


class CameraState(str, Enum):
    WAITING = "waiting"
    DRUM_STOP = "drum_stop"
    BURST_CAPTURE = "burst_capture"
    COOLDOWN = "cooldown"


@dataclass
class BurstEvent:
    cam_id: str
    ts: datetime
    frames: list[np.ndarray] = field(default_factory=list)


def open_capture(cfg: CameraConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg.device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cfg.id} at {cfg.device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    return cap


def crop_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def laplacian_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class CameraWorker:
    """Single-camera capture + motion-triggered burst logic.

    FSM: WAITING -> DRUM_STOP -> BURST_CAPTURE -> COOLDOWN -> WAITING
    """

    def __init__(
        self,
        cam_cfg: CameraConfig,
        motion_cfg: MotionConfig,
        burst_cfg: BurstConfig,
    ):
        self.cam_cfg = cam_cfg
        self.motion_cfg = motion_cfg
        self.burst_cfg = burst_cfg
        self.state = CameraState.WAITING
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=motion_cfg.history,
            varThreshold=motion_cfg.var_threshold,
            detectShadows=False,
        )
        self._quiet_since: float | None = None

    def probe(self) -> bool:
        """Quick test: can we grab a frame?"""
        cap = open_capture(self.cam_cfg)
        try:
            ok, frame = cap.read()
            return ok and frame is not None
        finally:
            cap.release()

    def run(self, out_queue) -> None:
        """Main capture loop. Pushes BurstEvent into out_queue."""
        raise NotImplementedError("Implemented in Stage 3 (CameraWorker)")
