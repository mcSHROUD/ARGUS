from __future__ import annotations

import numpy as np

from argus.config import BurstConfig, MotionConfig
from argus.motion_fsm import MotionFSM, State


def _make_fsm(stop_thr: float = 0.05, move_thr: float = 0.15) -> MotionFSM:
    motion = MotionConfig(
        drum_stop_threshold=stop_thr,
        drum_move_threshold=move_thr,
        stop_window_ms=50,
    )
    burst = BurstConfig(frames=3, max_duration_ms=500, min_sharpness=0.0)
    return MotionFSM(motion, burst, fps=60)


def _frame(value: int, shape: tuple[int, int, int] = (60, 80, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_emits_burst_after_quiet_window():
    fsm = _make_fsm()
    # warm up: two motion frames so FSM is "armed"
    for v in (0, 200):
        fsm.process(_frame(v))
    # quiet stretch: keep frames identical
    bursts = []
    for _ in range(20):
        b = fsm.process(_frame(200))
        if b is not None:
            bursts.append(b)
    assert len(bursts) == 1, f"expected exactly 1 burst, got {len(bursts)}"
    assert len(bursts[0].frames) == 3


def test_no_duplicate_bursts_without_motion_between():
    fsm = _make_fsm()
    # one motion + long quiet → 1 burst expected
    for v in (0, 200):
        fsm.process(_frame(v))
    bursts: list = []
    for _ in range(50):
        b = fsm.process(_frame(200))
        if b is not None:
            bursts.append(b)
    assert len(bursts) == 1, "without motion in between, only one burst per drum stop"


def test_two_bursts_with_motion_between():
    fsm = _make_fsm()
    bursts: list = []
    # cycle: motion → quiet → motion → quiet
    for cycle in range(2):
        # motion: alternating frames
        for v in (0, 200, 0, 200):
            b = fsm.process(_frame(v))
            if b:
                bursts.append(b)
        # quiet: same frame for many ticks
        for _ in range(15):
            b = fsm.process(_frame(200))
            if b:
                bursts.append(b)
    assert len(bursts) == 2, f"expected 2 bursts, got {len(bursts)}"


def test_state_transitions():
    fsm = _make_fsm()
    assert fsm.state == State.WAITING_FOR_STOP
    # one frame to seed
    fsm.process(_frame(0))
    # drum motion
    fsm.process(_frame(200))
    assert fsm.state == State.WAITING_FOR_STOP
    # quiet stretch
    for _ in range(10):
        fsm.process(_frame(200))
    # after burst is emitted FSM returns to WAITING_FOR_STOP
    assert fsm.state == State.WAITING_FOR_STOP


def test_roi_crop_applied():
    fsm = _make_fsm()
    big = _frame(0, shape=(200, 200, 3))
    big[50:100, 50:100] = 255  # paint a square inside ROI
    other = _frame(0, shape=(200, 200, 3))  # plain dark
    # process two different frames with ROI matching the painted square
    fsm.process(big, roi=(50, 50, 50, 50))
    burst = fsm.process(other, roi=(50, 50, 50, 50))
    # there's a big diff inside ROI; nothing emitted yet (need quiet window),
    # but no exception is the main check
    assert burst is None
