"""Orchestrator: ties cameras, detector, and storage together.

Two cameras run as independent threads producing BurstEvents into a shared
queue. The orchestrator pulls events, matches pairs by timestamp within a
sync window, runs inference, and persists results.

If only one camera produces a burst within the sync window, the orchestrator
still records the event with a `notes='unpaired'` flag — better to log a
half-observation than to drop a real cup.
"""
from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from argus.camera import BurstEvent, CameraWorker
from argus.config import Config
from argus.detector import DefectDetector
from argus.storage import Storage

logger = logging.getLogger(__name__)


def _make_heatmap_overlay(frame: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = frame.shape[:2]
    am = anomaly_map.squeeze().astype(np.float32)
    lo, hi = am.min(), am.max()
    if hi > lo:
        am = (am - lo) / (hi - lo)
    heatmap = cv2.applyColorMap((am * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (w, h))
    return cv2.addWeighted(frame, 1.0 - alpha, heatmap, alpha, 0)


@dataclass
class CupEvent:
    ts: datetime
    cam1_frames: list[np.ndarray] = field(default_factory=list)
    cam2_frames: list[np.ndarray] = field(default_factory=list)
    is_paired: bool = True

    @property
    def has_cam1(self) -> bool:
        return bool(self.cam1_frames)

    @property
    def has_cam2(self) -> bool:
        return bool(self.cam2_frames)


class BurstMatcher:
    """Buffers BurstEvents from 2 cameras and emits matched CupEvents.

    Strategy: keep the most recent unmatched burst from each camera. When a
    new burst arrives, try to pair it with the unmatched burst from the OTHER
    camera if they're within `window_ms`. If a buffered burst gets older than
    the window, flush it as an unpaired event.
    """

    def __init__(self, window_ms: int):
        self.window_s = window_ms / 1000.0
        self._pending: dict[str, BurstEvent] = {}
        self._lock = threading.Lock()

    def push(
        self, event: BurstEvent, cam_ids: tuple[str, str]
    ) -> tuple[CupEvent | None, CupEvent | None]:
        """Returns (matched_pair, evicted_unpaired).

        Matched_pair is set when this event paired with the buffered one from
        the other camera. Evicted_unpaired is set when this event displaced
        an older buffered event from the SAME camera — that older event is
        emitted as an unpaired cup so it isn't silently dropped.
        """
        cam1_id, cam2_id = cam_ids
        if event.cam_id not in cam_ids:
            logger.warning(f"unknown cam_id: {event.cam_id}")
            return None, None

        with self._lock:
            other_id = cam2_id if event.cam_id == cam1_id else cam1_id
            other = self._pending.get(other_id)

            if other is not None and abs((event.ts - other.ts).total_seconds()) <= self.window_s:
                self._pending.pop(other_id, None)
                cup = self._make_cup(event, other, cam1_id, cam2_id, paired=True)
                return cup, None

            evicted: CupEvent | None = None
            existing = self._pending.get(event.cam_id)
            if existing is not None:
                evicted = CupEvent(
                    ts=existing.ts,
                    cam1_frames=existing.frames if existing.cam_id == cam1_id else [],
                    cam2_frames=existing.frames if existing.cam_id != cam1_id else [],
                    is_paired=False,
                )
                logger.warning(
                    f"evicting unpaired {existing.cam_id} burst @ {existing.ts.isoformat()} "
                    f"(displaced by newer burst from same cam)"
                )
            self._pending[event.cam_id] = event
            return None, evicted

    def flush_stale(self, cam_ids: tuple[str, str]) -> list[CupEvent]:
        """Emit any buffered events that are older than the window."""
        cam1_id, _ = cam_ids
        out: list[CupEvent] = []
        now = datetime.utcnow()
        with self._lock:
            stale_ids = [
                cid
                for cid, ev in self._pending.items()
                if (now - ev.ts).total_seconds() > self.window_s
            ]
            for cid in stale_ids:
                ev = self._pending.pop(cid)
                cup = CupEvent(
                    ts=ev.ts,
                    cam1_frames=ev.frames if cid == cam1_id else [],
                    cam2_frames=ev.frames if cid != cam1_id else [],
                    is_paired=False,
                )
                out.append(cup)
        return out

    @staticmethod
    def _make_cup(
        a: BurstEvent, b: BurstEvent, cam1_id: str, cam2_id: str, paired: bool
    ) -> CupEvent:
        cam1, cam2 = (a, b) if a.cam_id == cam1_id else (b, a)
        return CupEvent(
            ts=cam1.ts,
            cam1_frames=cam1.frames,
            cam2_frames=cam2.frames,
            is_paired=paired,
        )


class Orchestrator:
    """Wires CameraWorkers, BurstMatcher, DefectDetector, and Storage together."""

    def __init__(
        self,
        config: Config,
        sources: tuple[str | int | None, str | int | None] = (None, None),
    ):
        self.config = config
        self.sources = sources
        self.storage = Storage(config.storage.db_path, config.storage.images_dir)
        self.detector = DefectDetector(config.detector)
        self.queue: queue.Queue[BurstEvent] = queue.Queue(maxsize=64)
        self.matcher = BurstMatcher(window_ms=config.sync.window_ms)
        self._stop_event = threading.Event()
        self._workers: list[CameraWorker] = []
        self._cam_ids = (config.cameras[0].id, config.cameras[1].id)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            self.detector.load()
            logger.info(
                f"detector loaded; threshold={getattr(self.detector, 'threshold', '?')}"
            )
        except FileNotFoundError as e:
            logger.error(f"{e}")
            self.storage.log_event("startup_error", {"reason": str(e)})
            return

        for cam_cfg, source in zip(self.config.cameras, self.sources, strict=True):
            # In replay mode with only one video, skip the camera with no source
            # rather than letting it bang on a missing /dev/video* device.
            replay_mode = any(isinstance(s, str) for s in self.sources)
            if replay_mode and source is None:
                logger.info(f"[{cam_cfg.id}] no replay source — skipping worker")
                continue
            worker = CameraWorker(
                cam_cfg=cam_cfg,
                motion_cfg=self.config.motion,
                burst_cfg=self.config.burst,
                out_queue=self.queue,
                source=source,
            )
            worker.start()
            self._workers.append(worker)
        if not self._workers:
            logger.error("no workers started")
            return

        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

        self.storage.log_event("orchestrator_start", {"cams": list(self._cam_ids)})
        logger.info(f"orchestrator running ({self._cam_ids[0]}, {self._cam_ids[1]})")

        try:
            self._main_loop()
        finally:
            self._shutdown()

    def _main_loop(self) -> None:
        last_flush = time.monotonic()
        while not self._stop_event.is_set():
            try:
                event = self.queue.get(timeout=0.5)
            except queue.Empty:
                if all(not w.is_alive() for w in self._workers):
                    logger.info("all workers exited; finishing")
                    break
            else:
                cup, evicted = self.matcher.push(event, self._cam_ids)
                if evicted is not None:
                    self._handle_cup(evicted)
                if cup is not None:
                    self._handle_cup(cup)

            # periodically flush stale unpaired bursts
            if time.monotonic() - last_flush > 1.0:
                last_flush = time.monotonic()
                for cup in self.matcher.flush_stale(self._cam_ids):
                    self._handle_cup(cup)

        # final flush after stop
        for cup in self.matcher.flush_stale(self._cam_ids):
            self._handle_cup(cup)

    def _handle_cup(self, cup: CupEvent) -> None:
        result = self.detector.infer(cup.cam1_frames, cup.cam2_frames)
        verdict = "defect" if result.is_defect else "good"

        cup_id = self.storage.insert_cup(
            ts=cup.ts,
            verdict=verdict,
            score=result.score,
            cam1_path=None,
            cam2_path=None,
            notes=None if cup.is_paired else "unpaired",
        )

        cam1_path: Path | None = None
        cam2_path: Path | None = None
        cam1_heatmap_path: Path | None = None
        cam2_heatmap_path: Path | None = None

        if cup.cam1_frames and result.worst_frame_idx_cam1 >= 0:
            worst1 = cup.cam1_frames[result.worst_frame_idx_cam1]
            cam1_path = self.storage.save_frame(worst1, "cam1", cup.ts, cup_id)
            if result.anomaly_map_cam1 is not None:
                cam1_heatmap_path = self.storage.save_frame(
                    _make_heatmap_overlay(worst1, result.anomaly_map_cam1),
                    "cam1_heatmap",
                    cup.ts,
                    cup_id,
                )

        if cup.cam2_frames and result.worst_frame_idx_cam2 >= 0:
            worst2 = cup.cam2_frames[result.worst_frame_idx_cam2]
            cam2_path = self.storage.save_frame(worst2, "cam2", cup.ts, cup_id)
            if result.anomaly_map_cam2 is not None:
                cam2_heatmap_path = self.storage.save_frame(
                    _make_heatmap_overlay(worst2, result.anomaly_map_cam2),
                    "cam2_heatmap",
                    cup.ts,
                    cup_id,
                )

        self._update_paths(cup_id, cam1_path, cam2_path, cam1_heatmap_path, cam2_heatmap_path)

        logger.info(
            f"cup #{cup_id}: {verdict} score={result.score:.2f} "
            f"paired={cup.is_paired} cam1={len(cup.cam1_frames)}f cam2={len(cup.cam2_frames)}f"
        )

    def _update_paths(
        self,
        cup_id: int,
        cam1: Path | None,
        cam2: Path | None,
        cam1_heatmap: Path | None = None,
        cam2_heatmap: Path | None = None,
    ) -> None:
        with self.storage.connection() as conn:
            conn.execute(
                "UPDATE cups SET cam1_path=?, cam2_path=?, cam1_heatmap_path=?, cam2_heatmap_path=? WHERE id=?",
                (
                    str(cam1) if cam1 else None,
                    str(cam2) if cam2 else None,
                    str(cam1_heatmap) if cam1_heatmap else None,
                    str(cam2_heatmap) if cam2_heatmap else None,
                    cup_id,
                ),
            )

    def _shutdown(self) -> None:
        logger.info("stopping workers")
        for w in self._workers:
            w.stop()
        for w in self._workers:
            w.join(timeout=3.0)
        self.storage.log_event(
            "orchestrator_stop",
            {
                "frames": [w.frames_seen for w in self._workers],
                "bursts": [w.bursts_emitted for w in self._workers],
            },
        )
        logger.info(
            "orchestrator stopped: "
            + ", ".join(
                f"{w.cam_cfg.id}={w.frames_seen}fr/{w.bursts_emitted}b" for w in self._workers
            )
        )
