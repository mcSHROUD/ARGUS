"""Frame-diff motion detector with finite-state machine for burst capture.

Used both for live camera workers and offline video extraction.

Physical model: a rotating drum stops at each position. Between stops the diff
metric (mean abs-diff between consecutive frames) is high; at rest it is low,
though never exactly zero because the cup keeps spinning from inertia. We need
to detect the *stops*, not the rotations, and capture a few frames during each
stop to get multiple views of the cup surface.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import cv2
import numpy as np

from argus.config import BurstConfig, MotionConfig

logger = logging.getLogger(__name__)


class State(str, Enum):
    WAITING_FOR_STOP = "waiting_for_stop"
    CAPTURING = "capturing"


@dataclass
class Burst:
    """A captured group of frames during one drum stop."""

    burst_idx: int
    start_frame: int
    end_frame: int
    ts: datetime = field(default_factory=datetime.utcnow)
    frames: list[np.ndarray] = field(default_factory=list)


def laplacian_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_diff_score(prev_small: np.ndarray, curr_small: np.ndarray) -> float:
    """Mean absolute pixel diff in [0, 1]."""
    return float(np.mean(cv2.absdiff(curr_small, prev_small))) / 255.0


class MotionFSM:
    """Tracks frame-to-frame motion, emits Burst events on drum stops.

    Usage::
        fsm = MotionFSM(motion_cfg, burst_cfg, fps=30)
        for raw_frame in stream:
            burst = fsm.process(raw_frame, roi)
            if burst is not None:
                # got a complete burst — handle it
                ...
    """

    DOWNSAMPLE = 4

    def __init__(self, motion_cfg: MotionConfig, burst_cfg: BurstConfig, fps: float):
        self.motion = motion_cfg
        self.burst_cfg = burst_cfg
        self.fps = fps

        self._stop_window = max(1, int(motion_cfg.stop_window_ms * fps / 1000))
        self._burst_max = max(1, int(burst_cfg.max_duration_ms * fps / 1000))

        self._state = State.WAITING_FOR_STOP
        self._quiet_count = 0
        self._armed = True  # require motion between bursts (anti-duplicate)
        self._prev_small: np.ndarray | None = None
        self._frame_idx = 0
        self._burst_idx = 0
        self._current: Burst | None = None

    @property
    def state(self) -> State:
        return self._state

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    def process(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> Burst | None:
        """Feed one frame. Returns a Burst when one finishes; None otherwise.

        Args:
            frame: BGR image from camera or video.
            roi: optional (x, y, w, h) crop applied before motion + storage.
        """
        if roi is not None:
            x, y, w, h = roi
            work = frame[y : y + h, x : x + w]
        else:
            work = frame

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            gray,
            (max(1, gray.shape[1] // self.DOWNSAMPLE), max(1, gray.shape[0] // self.DOWNSAMPLE)),
        )

        if self._prev_small is None:
            self._prev_small = small
            self._frame_idx += 1
            return None

        diff = frame_diff_score(self._prev_small, small)
        self._prev_small = small

        is_quiet = diff <= self.motion.drum_stop_threshold
        is_moving = diff >= self.motion.drum_move_threshold

        finished: Burst | None = None

        if self._state == State.WAITING_FOR_STOP:
            if is_moving:
                self._armed = True  # drum moved → next stop is a fresh cup
            if is_quiet:
                self._quiet_count += 1
                if self._quiet_count >= self._stop_window and self._armed:
                    self._state = State.CAPTURING
                    self._armed = False
                    self._current = Burst(
                        burst_idx=self._burst_idx,
                        start_frame=self._frame_idx,
                        end_frame=self._frame_idx,
                    )
                    self._try_append(work)
            else:
                self._quiet_count = 0

        elif self._state == State.CAPTURING:
            assert self._current is not None
            self._current.end_frame = self._frame_idx

            if is_moving:
                # next rotation started — finalize what we have
                if self._current.frames:
                    finished = self._current
                    self._burst_idx += 1
                self._current = None
                self._state = State.WAITING_FOR_STOP
                self._quiet_count = 0
            else:
                self._try_append(work)
                duration = self._current.end_frame - self._current.start_frame
                if (
                    len(self._current.frames) >= self.burst_cfg.frames
                    or duration >= self._burst_max
                ):
                    finished = self._current
                    self._burst_idx += 1
                    self._current = None
                    self._state = State.WAITING_FOR_STOP
                    self._quiet_count = 0

        self._frame_idx += 1
        return finished

    def _try_append(self, frame: np.ndarray) -> None:
        assert self._current is not None
        if laplacian_sharpness(frame) >= self.burst_cfg.min_sharpness:
            self._current.frames.append(frame.copy())
