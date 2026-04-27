from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from argus.camera import BurstEvent
from argus.pipeline import BurstMatcher


def _ev(cam: str, t: datetime) -> BurstEvent:
    return BurstEvent(cam_id=cam, ts=t, frames=[np.zeros((10, 10, 3), dtype=np.uint8)])


def test_pairs_within_window():
    m = BurstMatcher(window_ms=300)
    t = datetime(2026, 4, 23, 12, 0, 0)
    cup, evicted = m.push(_ev("cam1", t), ("cam1", "cam2"))
    assert cup is None and evicted is None
    cup, evicted = m.push(_ev("cam2", t + timedelta(milliseconds=150)), ("cam1", "cam2"))
    assert cup is not None and evicted is None
    assert cup.is_paired
    assert cup.has_cam1 and cup.has_cam2


def test_does_not_pair_outside_window():
    m = BurstMatcher(window_ms=200)
    t = datetime(2026, 4, 23, 12, 0, 0)
    cup, _ = m.push(_ev("cam1", t), ("cam1", "cam2"))
    assert cup is None
    cup, _ = m.push(_ev("cam2", t + timedelta(milliseconds=500)), ("cam1", "cam2"))
    assert cup is None  # too far apart


def test_flush_stale_emits_unpaired():
    m = BurstMatcher(window_ms=100)
    very_old = datetime.utcnow() - timedelta(seconds=5)
    m.push(_ev("cam1", very_old), ("cam1", "cam2"))
    flushed = m.flush_stale(("cam1", "cam2"))
    assert len(flushed) == 1
    assert not flushed[0].is_paired
    assert flushed[0].has_cam1 and not flushed[0].has_cam2


def test_pair_from_either_camera_first():
    m = BurstMatcher(window_ms=300)
    t = datetime(2026, 4, 23, 12, 0, 0)
    cup, _ = m.push(_ev("cam2", t), ("cam1", "cam2"))
    assert cup is None
    cup, _ = m.push(_ev("cam1", t + timedelta(milliseconds=100)), ("cam1", "cam2"))
    assert cup is not None
    assert cup.has_cam1 and cup.has_cam2


def test_evicts_old_same_cam_burst():
    m = BurstMatcher(window_ms=200)
    t = datetime(2026, 4, 23, 12, 0, 0)
    # cam1 sends two bursts in a row (cam2 silent)
    cup, evicted = m.push(_ev("cam1", t), ("cam1", "cam2"))
    assert cup is None and evicted is None
    # second cam1 burst displaces the first → first must come out as unpaired
    cup, evicted = m.push(_ev("cam1", t + timedelta(milliseconds=500)), ("cam1", "cam2"))
    assert cup is None
    assert evicted is not None
    assert not evicted.is_paired and evicted.has_cam1 and not evicted.has_cam2
