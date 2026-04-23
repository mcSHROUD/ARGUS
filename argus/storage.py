from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS cups (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    verdict     TEXT NOT NULL CHECK (verdict IN ('good', 'defect')),
    score       REAL NOT NULL,
    cam1_path   TEXT,
    cam2_path   TEXT,
    notes       TEXT
);
CREATE INDEX IF NOT EXISTS idx_cups_ts ON cups(ts);
CREATE INDEX IF NOT EXISTS idx_cups_verdict ON cups(verdict);

CREATE TABLE IF NOT EXISTS defect_regions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    cup_id   INTEGER NOT NULL REFERENCES cups(id) ON DELETE CASCADE,
    cam      TEXT NOT NULL,
    bbox     TEXT NOT NULL,
    score    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_regions_cup ON defect_regions(cup_id);

CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    cup_id       INTEGER NOT NULL REFERENCES cups(id) ON DELETE CASCADE,
    user_verdict TEXT NOT NULL CHECK (user_verdict IN ('confirmed', 'false_positive', 'missed')),
    ts           TEXT NOT NULL,
    comment      TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_cup ON feedback(cup_id);

CREATE TABLE IF NOT EXISTS system_events (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts    TEXT NOT NULL,
    kind  TEXT NOT NULL,
    data  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON system_events(ts);
"""


@dataclass
class CupRecord:
    id: int
    ts: datetime
    verdict: str
    score: float
    cam1_path: str | None
    cam2_path: str | None
    notes: str | None


@dataclass
class DefectRegion:
    cam: str
    bbox: tuple[int, int, int, int]
    score: float


class Storage:
    def __init__(self, db_path: Path, images_dir: Path):
        self.db_path = Path(db_path)
        self.images_dir = Path(images_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def save_frame(self, frame: np.ndarray, cam_id: str, ts: datetime, cup_id: int) -> Path:
        day = ts.strftime("%Y-%m-%d")
        out_dir = self.images_dir / day
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{cup_id:08d}_{cam_id}.jpg"
        path = out_dir / fname
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return path

    def insert_cup(
        self,
        ts: datetime,
        verdict: str,
        score: float,
        cam1_path: Path | None,
        cam2_path: Path | None,
        regions: list[DefectRegion] | None = None,
        notes: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO cups (ts, verdict, score, cam1_path, cam2_path, notes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ts.isoformat(),
                    verdict,
                    score,
                    str(cam1_path) if cam1_path else None,
                    str(cam2_path) if cam2_path else None,
                    notes,
                ),
            )
            cup_id = cur.lastrowid
            if regions:
                conn.executemany(
                    "INSERT INTO defect_regions (cup_id, cam, bbox, score) VALUES (?, ?, ?, ?)",
                    [(cup_id, r.cam, json.dumps(r.bbox), r.score) for r in regions],
                )
            return cup_id

    def log_event(self, kind: str, data: dict | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO system_events (ts, kind, data) VALUES (?, ?, ?)",
                (datetime.utcnow().isoformat(), kind, json.dumps(data) if data else None),
            )

    def list_cups(
        self,
        limit: int = 100,
        offset: int = 0,
        verdict: str | None = None,
    ) -> list[CupRecord]:
        query = "SELECT * FROM cups"
        params: list = []
        if verdict:
            query += " WHERE verdict = ?"
            params.append(verdict)
        query += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_cup(r) for r in rows]

    def get_cup(self, cup_id: int) -> CupRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cups WHERE id = ?", (cup_id,)).fetchone()
        return self._row_to_cup(row) if row else None

    def add_feedback(self, cup_id: int, user_verdict: str, comment: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (cup_id, user_verdict, ts, comment) VALUES (?, ?, ?, ?)",
                (cup_id, user_verdict, datetime.utcnow().isoformat(), comment),
            )

    @staticmethod
    def _row_to_cup(row: sqlite3.Row) -> CupRecord:
        return CupRecord(
            id=row["id"],
            ts=datetime.fromisoformat(row["ts"]),
            verdict=row["verdict"],
            score=row["score"],
            cam1_path=row["cam1_path"],
            cam2_path=row["cam2_path"],
            notes=row["notes"],
        )
