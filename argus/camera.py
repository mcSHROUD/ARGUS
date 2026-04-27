"""Live camera capture worker.

Each `CameraWorker` runs in its own thread, reads frames from a V4L2 device
(or any cv2.VideoCapture source), feeds them to a MotionFSM, and pushes
completed bursts to a shared queue for the orchestrator.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from argus.config import BurstConfig, CameraConfig, MotionConfig
from argus.motion_fsm import MotionFSM

logger = logging.getLogger(__name__)


@dataclass
class BurstEvent:
    cam_id: str
    ts: datetime
    frames: list[np.ndarray] = field(default_factory=list)
    burst_idx: int = 0


def open_capture(cfg: CameraConfig, source: str | int | None = None) -> cv2.VideoCapture:
    """Open a V4L2 device by default, or any cv2-compatible source if given.

    `source` can be a video file path for replay/testing.
    """
    if source is None:
        source = cfg.device
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cfg.id} at {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    return cap


class CameraWorker(threading.Thread):
    """One worker per camera. Pushes BurstEvent to `out_queue`."""

    def __init__(
        self,
        cam_cfg: CameraConfig,
        motion_cfg: MotionConfig,
        burst_cfg: BurstConfig,
        out_queue: queue.Queue[BurstEvent],
        source: str | int | None = None,
        retry_delay_s: float = 2.0,
        max_open_failures: int = 3,
    ):
        super().__init__(name=f"cam-{cam_cfg.id}", daemon=True)
        self.cam_cfg = cam_cfg
        self.motion_cfg = motion_cfg
        self.burst_cfg = burst_cfg
        self.out_queue = out_queue
        self.source = source
        self.retry_delay_s = retry_delay_s
        self.max_open_failures = max_open_failures
        self._stop_event = threading.Event()
        self._exhausted = False  # set when capture source had no more frames (e.g. EOF)
        self._open_failures = 0
        self._frames_seen = 0
        self._bursts_emitted = 0

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    @property
    def bursts_emitted(self) -> int:
        return self._bursts_emitted

    def probe(self) -> bool:
        cap = open_capture(self.cam_cfg, self.source)
        try:
            ok, frame = cap.read()
            return ok and frame is not None
        finally:
            cap.release()

    def run(self) -> None:
        logger.info(f"[{self.cam_cfg.id}] starting worker (source={self.source or self.cam_cfg.device})")
        while not self._stop_event.is_set():
            try:
                self._capture_loop()
                self._open_failures = 0  # successful open resets the counter
            except RuntimeError as e:
                self._open_failures += 1
                logger.error(
                    f"[{self.cam_cfg.id}] open failed ({self._open_failures}/{self.max_open_failures}): {e}"
                )
                if self._open_failures >= self.max_open_failures:
                    logger.error(f"[{self.cam_cfg.id}] giving up after {self._open_failures} failures")
                    break
                if self._stop_event.wait(self.retry_delay_s):
                    break
            except Exception as e:  # noqa: BLE001
                logger.exception(f"[{self.cam_cfg.id}] capture loop crashed: {e}")
                if self._stop_event.wait(self.retry_delay_s):
                    break
            if self._exhausted:
                logger.info(f"[{self.cam_cfg.id}] source exhausted, exiting")
                break
        logger.info(f"[{self.cam_cfg.id}] worker stopped")

    def _capture_loop(self) -> None:
        cap = open_capture(self.cam_cfg, self.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or float(self.cam_cfg.fps)
        fsm = MotionFSM(self.motion_cfg, self.burst_cfg, fps=fps)
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if isinstance(self.source, str):
                        # file source: EOF means we're done, don't retry
                        self._exhausted = True
                    else:
                        logger.warning(f"[{self.cam_cfg.id}] capture failed; retrying")
                    return
                self._frames_seen += 1
                burst = fsm.process(frame, self.cam_cfg.roi)
                if burst is not None and burst.frames:
                    self._bursts_emitted += 1
                    event = BurstEvent(
                        cam_id=self.cam_cfg.id,
                        ts=burst.ts,
                        frames=burst.frames,
                        burst_idx=burst.burst_idx,
                    )
                    self.out_queue.put(event)
                    logger.info(
                        f"[{self.cam_cfg.id}] burst #{burst.burst_idx}: "
                        f"{len(burst.frames)} frames @ {burst.ts.isoformat(timespec='milliseconds')}"
                    )
        finally:
            cap.release()


def laplacian_sharpness(frame: np.ndarray) -> float:
    """Re-export for backward compat with extract.py callers."""
    from argus.motion_fsm import laplacian_sharpness as _lap

    return _lap(frame)
