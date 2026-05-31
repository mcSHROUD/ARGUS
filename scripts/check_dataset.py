#!/usr/bin/env python3
"""
Evaluate dataset sufficiency:
1. Find optimal threshold using real defect data (F1)
2. K-fold check: score held-out normal frames to detect gaps in training coverage
3. Give concrete recommendation
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.models import Patchcore
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).parent.parent
CKPT = ROOT / "models_new" / "patchcore.ckpt"
THR_FILE = ROOT / "models_new" / "patchcore.threshold"
GOOD_DIR = ROOT / "data" / "raw" / "good" / "cam1"
REAL_DIR = ROOT / "data" / "raw" / "defects_real"
IMAGE_SIZE = (256, 256)


def load_model() -> Patchcore:
    model = Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), pre_trained=False)
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def preprocess(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    return torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0


def score_files(model: Patchcore, files: list[Path], label: str = "") -> np.ndarray:
    scores = []
    for i, f in enumerate(files):
        t = preprocess(f).unsqueeze(0)
        with torch.no_grad():
            out = model.model(t)
        scores.append(float(out.pred_score.item()))
        if label and (i + 1) % 50 == 0:
            print(f"  {label}: {i+1}/{len(files)}", flush=True)
    return np.array(scores)


def find_optimal_threshold(normal: np.ndarray, defects: np.ndarray) -> tuple[float, float]:
    y_true = [0] * len(normal) + [1] * len(defects)
    y_scores = normal.tolist() + defects.tolist()
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best]), float(f1[best])


def kfold_check(model: Patchcore, files: list[Path], threshold: float, k: int = 5) -> dict:
    """Score normal frames using a model trained on all 155. Frames are in the memory bank,
    so scores are optimistic — but high scores still signal risky cups."""
    scores = score_files(model, files, "normal")
    fpr_per_fold = []
    near_threshold = []

    fold_size = len(files) // k
    for fold in range(k):
        start, end = fold * fold_size, (fold + 1) * fold_size
        fold_scores = scores[start:end]
        fp = (fold_scores > threshold).sum()
        fpr_per_fold.append(fp / len(fold_scores))
        risky = fold_scores[fold_scores > threshold - 2.0]
        near_threshold.extend(risky.tolist())

    return {
        "scores": scores,
        "fpr_per_fold": fpr_per_fold,
        "near_threshold": near_threshold,
    }


def main() -> None:
    if not CKPT.exists():
        sys.exit(f"Checkpoint not found: {CKPT}")

    print("Loading model...")
    model = load_model()

    good_files = sorted(GOOD_DIR.glob("*.jpg"))
    defect_files = []
    for subdir in sorted(REAL_DIR.iterdir()):
        if subdir.is_dir():
            defect_files.extend(sorted(subdir.glob("*.jpg")))

    print(f"Scoring {len(good_files)} normal frames...")
    normal_scores = score_files(model, good_files, "normal")

    print(f"Scoring {len(defect_files)} real defect photos...")
    defect_scores = score_files(model, defect_files, "defect")

    # --- Optimal threshold ---
    opt_thr, best_f1 = find_optimal_threshold(normal_scores, defect_scores)
    tp = int((defect_scores > opt_thr).sum())
    fp = int((normal_scores > opt_thr).sum())
    recall = tp / len(defect_scores)
    fpr = fp / len(normal_scores)

    print(f"\n{'='*55}")
    print(f"  OPTIMAL THRESHOLD (max F1 on real data)")
    print(f"{'='*55}")
    print(f"  Threshold  : {opt_thr:.4f}  (was: 45.3182)")
    print(f"  F1 score   : {best_f1:.3f}")
    print(f"  Recall     : {tp}/{len(defect_scores)} ({100*recall:.0f}%)")
    print(f"  FPR        : {fp}/{len(normal_scores)} ({100*fpr:.1f}%)")

    # --- Score gap analysis ---
    gap = defect_scores.min() - normal_scores.max()
    n_std = gap / normal_scores.std()
    print(f"\n{'='*55}")
    print(f"  SCORE GAP ANALYSIS")
    print(f"{'='*55}")
    print(f"  Normal : min={normal_scores.min():.2f}  median={np.median(normal_scores):.2f}  "
          f"max={normal_scores.max():.2f}  std={normal_scores.std():.2f}")
    print(f"  Defects: min={defect_scores.min():.2f}  median={np.median(defect_scores):.2f}  "
          f"max={defect_scores.max():.2f}")
    print(f"  Gap (min_defect - max_normal): {gap:+.2f} pts  ({n_std:.1f}x std)")

    # --- Near-threshold frames (risky normal cups) ---
    risky = normal_scores[normal_scores > opt_thr - 3.0]
    print(f"\n  Normal frames within 3 pts of threshold ({opt_thr:.1f}):")
    print(f"  {len(risky)}/{len(normal_scores)} frames  —  max={normal_scores.max():.2f}")

    # --- Recommendation ---
    print(f"\n{'='*55}")
    print(f"  RECOMMENDATION")
    print(f"{'='*55}")
    if gap > 3.0:
        verdict = "SUFFICIENT"
        detail = "Gap > 3 pts — robust even with new cup variations."
    elif gap > 1.0:
        verdict = "BORDERLINE"
        detail = f"Gap = {gap:.1f} pts ({n_std:.1f} sigma). Works now but may produce false positives on new lighting/cup angles."
    else:
        verdict = "INSUFFICIENT"
        detail = "Gap < 1 pt — high FPR risk in production."

    print(f"  Dataset: {verdict}")
    print(f"  {detail}")

    if verdict != "SUFFICIENT":
        needed = max(0, int(300 - len(good_files)))
        print(f"\n  Action: collect ~{needed} more normal frames (target: 300 total).")
        print(f"  Vary lighting, cup angles, fill levels if possible.")

    # --- Save new threshold ---
    THR_FILE.write_text(f"{opt_thr:.6f}\n")
    print(f"\n  Threshold updated: {opt_thr:.4f} -> saved to {THR_FILE.name}")


if __name__ == "__main__":
    main()
