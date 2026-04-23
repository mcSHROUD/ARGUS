from __future__ import annotations

from datetime import datetime

import numpy as np

from argus.storage import DefectRegion, Storage


def test_insert_and_list(tmp_path):
    storage = Storage(tmp_path / "argus.db", tmp_path / "img")
    now = datetime.utcnow()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cup_id = storage.insert_cup(
        ts=now,
        verdict="defect",
        score=0.87,
        cam1_path=storage.save_frame(frame, "cam1", now, 1),
        cam2_path=storage.save_frame(frame, "cam2", now, 1),
        regions=[DefectRegion(cam="cam1", bbox=(10, 10, 20, 20), score=0.9)],
    )

    cups = storage.list_cups()
    assert len(cups) == 1
    assert cups[0].id == cup_id
    assert cups[0].verdict == "defect"
    assert cups[0].score == 0.87


def test_feedback(tmp_path):
    storage = Storage(tmp_path / "argus.db", tmp_path / "img")
    now = datetime.utcnow()
    cup_id = storage.insert_cup(now, "defect", 0.9, None, None)
    storage.add_feedback(cup_id, "confirmed", "reviewed by operator")

    with storage.connection() as conn:
        row = conn.execute("SELECT * FROM feedback WHERE cup_id = ?", (cup_id,)).fetchone()
    assert row["user_verdict"] == "confirmed"
