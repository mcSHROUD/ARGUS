#!/usr/bin/env python3
"""Sanity-test the trained detector: normal frames should score below threshold,
synthetically damaged frames should score above."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/app")
from argus.config import load_config
from argus.detector import DefectDetector


def main() -> None:
    cfg = load_config("/app/config.yaml")
    det = DefectDetector(cfg.detector)
    det.load()

    frames_dir = Path("/data/raw/good/cam1")
    files = sorted(frames_dir.glob("*.jpg"))[:5]
    frames = [cv2.imread(str(f)) for f in files]

    print(f"threshold: {det.threshold:.2f}\n")

    print("=== inference on 5 TRAINING frames (expected: good) ===")
    result = det.infer(frames, [])
    print(f"per-frame scores: {[round(s, 2) for s in result.per_frame_scores_cam1]}")
    print(f"max score: {result.score:.2f} | is_defect: {result.is_defect}")

    print("\n=== inference on frame with big painted blob (expected: defect) ===")
    noisy = frames[0].copy()
    cv2.rectangle(noisy, (100, 100), (500, 500), (0, 255, 255), -1)
    result = det.infer([noisy], [])
    print(f"score: {result.score:.2f} | is_defect: {result.is_defect}")

    print("\n=== inference on frame with gaussian noise (borderline) ===")
    noise = np.random.randint(0, 60, frames[0].shape, dtype=np.int16)
    noisy = np.clip(frames[0].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    result = det.infer([noisy], [])
    print(f"score: {result.score:.2f} | is_defect: {result.is_defect}")


if __name__ == "__main__":
    main()
