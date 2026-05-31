#!/usr/bin/env python3
"""Score real defect photos through PatchCore and report results."""
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
REAL_DIR = ROOT / "data" / "raw" / "defects_real"
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
    if img is None:
        raise ValueError(f"Cannot read {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    return torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0


def score_files(model: Patchcore, files: list[Path]) -> list[float]:
    scores = []
    for f in files:
        t = preprocess(f).unsqueeze(0)
        with torch.no_grad():
            out = model.model(t)
        scores.append(float(out.pred_score.item()))
    return scores


def main() -> None:
    if not CKPT.exists():
        sys.exit(f"Checkpoint not found: {CKPT}")

    print(f"Loading model...")
    model, threshold = load_model()
    print(f"Threshold: {threshold:.4f}\n")

    # Score normal frames for reference
    good_files = sorted(GOOD_DIR.glob("*.jpg"))
    good_scores = np.array(score_files(model, good_files))
    print(f"Normal frames (n={len(good_scores)}): "
          f"median={np.median(good_scores):.2f}  max={good_scores.max():.2f}  "
          f"FPR={100*(good_scores > threshold).mean():.1f}%")

    # Score each defect type subfolder
    defect_subdirs = sorted(d for d in REAL_DIR.iterdir() if d.is_dir())
    if not defect_subdirs:
        sys.exit(f"No subfolders found in {REAL_DIR}")

    print()
    all_defect_scores = []
    for subdir in defect_subdirs:
        files = sorted(subdir.glob("*.jpg")) + sorted(subdir.glob("*.png"))
        if not files:
            continue
        scores = score_files(model, files)
        arr = np.array(scores)
        all_defect_scores.extend(scores)
        above = int((arr > threshold).sum())
        print(f"=== {subdir.name.upper()} (n={len(scores)}) ===")
        print(f"  detected : {above}/{len(scores)} ({100*above/len(scores):.0f}%)")
        print(f"  min / median / max : {arr.min():.2f} / {np.median(arr):.2f} / {arr.max():.2f}")
        print(f"  per-file scores:")
        for f, s in zip(files, scores):
            flag = "DEFECT" if s > threshold else "missed"
            print(f"    [{flag}]  {s:6.2f}  {f.name}")
        print()

    if all_defect_scores:
        arr = np.array(all_defect_scores)
        above = int((arr > threshold).sum())
        print(f"=== TOTAL real defects: {above}/{len(arr)} detected ({100*above/len(arr):.0f}%) ===")


if __name__ == "__main__":
    main()
