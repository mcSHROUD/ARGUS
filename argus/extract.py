from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from argus.camera import laplacian_sharpness
from argus.config import BurstConfig, MotionConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractedBurst:
    burst_idx: int
    start_frame: int
    end_frame: int
    frames: list[np.ndarray]


def _frame_diff(prev_gray_small: np.ndarray, curr_gray_small: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(curr_gray_small, prev_gray_small))) / 255.0


def extract_bursts(
    video_path: Path,
    roi: tuple[int, int, int, int] | None,
    motion_cfg: MotionConfig,
    burst_cfg: BurstConfig,
) -> list[ExtractedBurst]:
    """Detect drum-stops in video via frame-to-frame diff, extract burst frames.

    Physical model: drum rotates (high inter-frame diff), stops (low diff), cup settles
    briefly. Each detected stop → one burst.

    FSM:
        INIT → wait until quiet for `stop_window` frames → CAPTURING
        CAPTURING → accumulate sharp frames until either:
                    - burst.frames count reached
                    - motion spikes above `drum_move_threshold` (next rotation)
                    → WAITING (for next stop cycle)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"video: {video_path} | fps={fps:.1f} | frames={total}")

    stop_window_frames = max(1, int(motion_cfg.stop_window_ms * fps / 1000))
    burst_max_frames = max(1, int(burst_cfg.max_duration_ms * fps / 1000))
    stop_thr = motion_cfg.drum_stop_threshold
    move_thr = motion_cfg.drum_move_threshold

    bursts: list[ExtractedBurst] = []
    state = "WAITING_FOR_STOP"
    quiet_count = 0
    current: ExtractedBurst | None = None
    prev_small: np.ndarray | None = None
    frame_idx = 0
    # After a burst we must see rotation (diff >= move_thr) before we arm the next stop
    # detection. Otherwise the remaining quiet tail of the same stop re-triggers CAPTURING.
    armed = True

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ROI crop for both diff measurement and saved frames
        if roi is not None:
            x, y, w, h = roi
            work = frame[y : y + h, x : x + w]
        else:
            work = frame

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (gray.shape[1] // 4, gray.shape[0] // 4))

        if prev_small is None:
            prev_small = small
            frame_idx += 1
            continue

        diff = _frame_diff(prev_small, small)
        prev_small = small
        is_quiet = diff <= stop_thr
        is_moving = diff >= move_thr

        if state == "WAITING_FOR_STOP":
            if is_moving:
                armed = True
            if is_quiet:
                quiet_count += 1
                if quiet_count >= stop_window_frames and armed:
                    state = "CAPTURING"
                    armed = False
                    current = ExtractedBurst(
                        burst_idx=len(bursts),
                        start_frame=frame_idx,
                        end_frame=frame_idx,
                        frames=[],
                    )
                    _try_append(current, work, burst_cfg)
            else:
                quiet_count = 0

        elif state == "CAPTURING":
            assert current is not None
            current.end_frame = frame_idx

            if is_moving:
                if current.frames:
                    bursts.append(current)
                    logger.info(
                        f"burst #{current.burst_idx}: frames "
                        f"{current.start_frame}-{current.end_frame} "
                        f"({len(current.frames)} kept) — ended by motion"
                    )
                current = None
                state = "WAITING_FOR_STOP"
                quiet_count = 0
            else:
                _try_append(current, work, burst_cfg)
                duration = current.end_frame - current.start_frame
                if len(current.frames) >= burst_cfg.frames or duration >= burst_max_frames:
                    bursts.append(current)
                    logger.info(
                        f"burst #{current.burst_idx}: frames "
                        f"{current.start_frame}-{current.end_frame} "
                        f"({len(current.frames)} kept) — full"
                    )
                    current = None
                    state = "WAITING_FOR_STOP"
                    quiet_count = 0

        frame_idx += 1

    cap.release()
    logger.info(f"done: {len(bursts)} bursts from {frame_idx} frames")
    return bursts


def _try_append(burst: ExtractedBurst, frame: np.ndarray, cfg: BurstConfig) -> None:
    if laplacian_sharpness(frame) >= cfg.min_sharpness:
        burst.frames.append(frame.copy())


def save_bursts(bursts: list[ExtractedBurst], out_dir: Path, prefix: str = "burst") -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for b in bursts:
        for k, frame in enumerate(b.frames):
            path = out_dir / f"{prefix}_{b.burst_idx:04d}_frame_{k:02d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            total += 1
    return total
