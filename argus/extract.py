"""Offline burst extraction from a recorded video file.

Used to build the calibration dataset before cameras are mounted, and as the
backend of `argus replay` for end-to-end testing.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2

from argus.config import BurstConfig, MotionConfig
from argus.motion_fsm import Burst, MotionFSM

logger = logging.getLogger(__name__)


def extract_bursts(
    video_path: Path,
    roi: tuple[int, int, int, int] | None,
    motion_cfg: MotionConfig,
    burst_cfg: BurstConfig,
) -> list[Burst]:
    """Read a video, run MotionFSM, return the list of completed bursts."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"video: {video_path} | fps={fps:.1f} | frames={total}")

    fsm = MotionFSM(motion_cfg, burst_cfg, fps=fps)
    bursts: list[Burst] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        burst = fsm.process(frame, roi)
        if burst is not None:
            bursts.append(burst)
            logger.info(
                f"burst #{burst.burst_idx}: frames {burst.start_frame}-{burst.end_frame} "
                f"({len(burst.frames)} kept)"
            )

    cap.release()
    logger.info(f"done: {len(bursts)} bursts from {fsm.frame_idx} frames")
    return bursts


def save_bursts(bursts: list[Burst], out_dir: Path, prefix: str = "burst") -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for b in bursts:
        for k, frame in enumerate(b.frames):
            path = out_dir / f"{prefix}_{b.burst_idx:04d}_frame_{k:02d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            total += 1
    return total
