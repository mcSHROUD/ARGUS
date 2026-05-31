#!/usr/bin/env python3
"""Score normal frames and synthetic defects through PatchCore, print recall/FPR stats."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.models import Patchcore

ROOT = Path(__file__).parent.parent
CKPT = ROOT / "models_new" / "patchcore.ckpt"
THR_FILE = ROOT / "models_new" / "patchcore.threshold"
GOOD_DIR = ROOT / "data" / "raw" / "good" / "cam1"
DEFECT_DIR = ROOT / "data" / "raw" / "defects" / "cam1"
IMAGE_SIZE = (256, 256)


def load_model() -> tuple[Patchcore, float]:
    model = Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), pre_trained=False)
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    threshold = float(THR_FILE.read_text().strip()) if THR_FILE.exists() else 0.5
    return model, threshold


def preprocess(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    return torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0


def score_files(model: Patchcore, files: list[Path], label: str) -> list[float]:
    scores = []
    for i, f in enumerate(files):
        t = preprocess(f).unsqueeze(0)
        with torch.no_grad():
            out = model.model(t)
        s = float(out.pred_score.item())
        scores.append(s)
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"  {label}: {i+1}/{len(files)} done...", flush=True)
    return scores


def report(name: str, scores: list[float], threshold: float) -> None:
    arr = np.array(scores)
    above = int((arr > threshold).sum())
    print(f"\n{'='*50}")
    print(f"  {name}  (n={len(scores)})")
    print(f"{'='*50}")
    print(f"  threshold : {threshold:.4f}")
    print(f"  detected  : {above}/{len(scores)}  ({100*above/len(scores):.1f}%)")
    print(f"  min / p25 / median / p75 / p95 / max")
    print(f"  {arr.min():.3f} / {np.percentile(arr,25):.3f} / {np.median(arr):.3f} / "
          f"{np.percentile(arr,75):.3f} / {np.percentile(arr,95):.3f} / {arr.max():.3f}")


def main() -> None:
    if not CKPT.exists():
        sys.exit(f"Checkpoint not found: {CKPT}")

    good_files = sorted(GOOD_DIR.glob("*.jpg"))
    defect_files = sorted(DEFECT_DIR.glob("*.jpg"))

    if not good_files:
        sys.exit(f"No images in {GOOD_DIR}")
    if not defect_files:
        sys.exit(f"No images in {DEFECT_DIR}")

    print(f"Loading model from {CKPT}...")
    model, threshold = load_model()
    print(f"Threshold: {threshold:.4f}")
    print(f"\nScoring {len(good_files)} normal frames...")
    good_scores = score_files(model, good_files, "good")
    print(f"\nScoring {len(defect_files)} synthetic defect frames...")
    defect_scores = score_files(model, defect_files, "defect")

    report("NORMAL frames (expected: below threshold)", good_scores, threshold)
    report("SYNTHETIC DEFECTS (expected: above threshold)", defect_scores, threshold)

    good_arr = np.array(good_scores)
    def_arr = np.array(defect_scores)
    overlap = (def_arr < threshold).sum()
    print(f"\n>>> MISSED defects : {overlap}/{len(defect_scores)}  "
          f"({100*overlap/len(defect_scores):.1f}%)")
    fp = (good_arr > threshold).sum()
    print(f">>> FALSE positives: {fp}/{len(good_scores)}  "
          f"({100*fp/len(good_scores):.1f}%)")

    best_thr = find_best_threshold(good_arr, def_arr)
    print(f"\n>>> Suggested threshold (max F1): {best_thr:.4f}")

    print("\n--- Score breakdown by defect type ---")
    by_type: dict[str, list[float]] = {}
    for f, s in zip(defect_files, defect_scores):
        dtype = f.stem.split("_")[0]
        by_type.setdefault(dtype, []).append(s)
    for dtype, type_scores in sorted(by_type.items()):
        arr = np.array(type_scores)
        above = int((arr > threshold).sum())
        print(f"  {dtype:10s}: detected {above}/{len(type_scores)} ({100*above/len(type_scores):.0f}%)  "
              f"median={np.median(arr):.2f}  max={arr.max():.2f}")


def find_best_threshold(good: np.ndarray, defect: np.ndarray) -> float:
    all_scores = np.concatenate([good, defect])
    candidates = np.linspace(all_scores.min(), all_scores.max(), 500)
    best_f1, best_t = 0.0, candidates[0]
    for t in candidates:
        tp = (defect > t).sum()
        fp = (good > t).sum()
        fn = (defect <= t).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


if __name__ == "__main__":
    main()
