#!/usr/bin/env python3
"""Diagnose motion profile of a video using frame-to-frame absolute diff.
Drum rotation → high diff; drum stopped → low diff."""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--downsample", type=int, default=4, help="Downscale factor for speed")
    ap.add_argument("--print-every", type=int, default=1)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {args.video} | {w}x{h} @ {fps:.1f}fps | {n_frames} frames")
    print(f"metric: normalized mean abs-diff between consecutive (grayscale, downsample x{args.downsample}) frames\n")

    prev: np.ndarray | None = None
    diffs: list[float] = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            gray, (gray.shape[1] // args.downsample, gray.shape[0] // args.downsample)
        )
        if prev is not None:
            diff = float(np.mean(cv2.absdiff(small, prev))) / 255.0
            diffs.append(diff)
            if idx % args.print_every == 0:
                bar = "#" * int(diff * 1000)
                print(f"frame {idx:4d} ({idx/fps:5.2f}s): {diff:.4f} {bar}")
        prev = small
        idx += 1
    cap.release()

    arr = np.array(diffs)
    print("\n=== stats ===")
    print(f"frames: {len(arr)}")
    for p in (5, 10, 25, 50, 75, 90, 95):
        print(f"  p{p:02d}: {np.percentile(arr, p):.4f}")
    print(f"  max: {arr.max():.4f}")

    low = np.percentile(arr, 25)
    high = np.percentile(arr, 75)
    print("\n=== suggested ===")
    print(f"drum_stop_threshold (diff low, <= this = quiet): {low:.4f}")
    print(f"drum_move_threshold (diff high, >= this = rotating): {high:.4f}")


if __name__ == "__main__":
    main()
