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
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT NOT NULL,
    verdict            TEXT NOT NULL CHECK (verdict IN ('good', 'defect')),
    score              REAL NOT NULL,
    cam1_path          TEXT,
    cam2_path          TEXT,
    cam1_heatmap_path  TEXT,
    cam2_heatmap_path  TEXT,
    notes              TEXT
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
    cam1_heatmap_path: str | None
    cam2_heatmap_path: str | None
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
            # lazy migration: add heatmap columns if DB was created before this feature
            for col in ("cam1_heatmap_path", "cam2_heatmap_path"):
                try:
                    conn.execute(f"ALTER TABLE cups ADD COLUMN {col} TEXT")
                except sqlite3.OperationalError:
                    pass  # column already exists

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
        date: str | None = None,
    ) -> list[CupRecord]:
        query = "SELECT * FROM cups WHERE 1=1"
        params: list = []
        if verdict:
            query += " AND verdict = ?"
            params.append(verdict)
        if date:
            query += " AND date(ts) = ?"
            params.append(date)
        query += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_cup(r) for r in rows]

    def get_daily_stats(self, date: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN verdict='defect' THEN 1 ELSE 0 END) as defects,
                    SUM(CASE WHEN verdict='good'   THEN 1 ELSE 0 END) as good
                FROM cups WHERE date(ts) = ?""",
                (date,),
            ).fetchone()
        total = row["total"] or 0
        defects = row["defects"] or 0
        good = row["good"] or 0
        return {
            "total": total,
            "defects": defects,
            "good": good,
            "defect_rate": (defects / total * 100) if total > 0 else 0.0,
        }

    def get_hourly_counts(self, date: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT
                    CAST(strftime('%H', ts) AS INTEGER) as hour,
                    SUM(CASE WHEN verdict='defect' THEN 1 ELSE 0 END) as defects,
                    SUM(CASE WHEN verdict='good'   THEN 1 ELSE 0 END) as good
                FROM cups WHERE date(ts) = ?
                GROUP BY hour ORDER BY hour""",
                (date,),
            ).fetchall()
        return [{"hour": r["hour"], "defects": r["defects"], "good": r["good"]} for r in rows]

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
            cam1_heatmap_path=row["cam1_heatmap_path"],
            cam2_heatmap_path=row["cam2_heatmap_path"],
            notes=row["notes"],
        )
