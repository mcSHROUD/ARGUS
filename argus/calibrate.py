from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

from argus.config import Config

logger = logging.getLogger(__name__)


def train(
    config: Config,
    normal_dir: Path,
    abnormal_dir: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Train PatchCore on normal cup frames. Returns path to saved checkpoint.

    Args:
        normal_dir: folder of 'good' cup images (JPEG/PNG)
        abnormal_dir: optional folder of defective images for validation
        out_dir: where to write checkpoint + logs (default: results/)
    """
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not normal_dir.exists() or not any(normal_dir.iterdir()):
        raise RuntimeError(f"normal_dir is empty or missing: {normal_dir}")

    n_normal = len(list(normal_dir.glob("*.jpg"))) + len(list(normal_dir.glob("*.png")))
    logger.info(f"training on {n_normal} normal frames from {normal_dir}")
    if abnormal_dir and abnormal_dir.exists():
        n_abn = len(list(abnormal_dir.glob("*.jpg"))) + len(list(abnormal_dir.glob("*.png")))
        logger.info(f"validation: {n_abn} abnormal frames from {abnormal_dir}")

    datamodule = Folder(
        name="cups",
        root=normal_dir.parent.parent,
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir if abnormal_dir and abnormal_dir.exists() else None,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=2,
        # PatchCore needs a small coreset; with few samples we go with defaults
        val_split_mode="same_as_test" if not abnormal_dir else "from_test",
        seed=42,
    )

    model = Patchcore(
        backbone=config.detector.backbone,
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
    )

    engine = Engine(
        default_root_dir=str(out_dir),
        accelerator="cpu",
        devices=1,
        max_epochs=1,  # PatchCore is memory-bank based, one pass is enough
    )
    engine.fit(model=model, datamodule=datamodule)

    ckpt_path = out_dir / "patchcore.ckpt"
    engine.trainer.save_checkpoint(str(ckpt_path))
    logger.info(f"checkpoint saved: {ckpt_path}")

    # Compute scores on the training set to pick a percentile-based threshold.
    # Without abnormal data we default to p99 of normal scores + small margin.
    threshold = _compute_percentile_threshold(model, datamodule, p=0.99, margin=0.1)
    logger.info(f"percentile(0.99) threshold: {threshold:.4f}")

    _write_threshold(threshold, ckpt_path)
    return ckpt_path


def _compute_percentile_threshold(
    model,
    datamodule,
    p: float = 0.99,
    margin: float = 0.1,
) -> float:
    """Run inference on the training set, return percentile(p) * (1 + margin)."""
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for batch in datamodule.train_dataloader():
            image = batch.image if hasattr(batch, "image") else batch
            out = model.model(image)
            if hasattr(out, "pred_score") and out.pred_score is not None:
                scores.extend(out.pred_score.cpu().numpy().tolist())
    if not scores:
        logger.warning("no training scores collected, defaulting threshold=0.5")
        return 0.5
    thr = float(np.percentile(scores, p * 100) * (1.0 + margin))
    return thr


def _write_threshold(threshold: float, ckpt_path: Path) -> None:
    """Save tuned threshold next to the checkpoint.

    DefectDetector will load this automatically if present, overriding config.
    """
    thr_path = ckpt_path.with_suffix(".threshold")
    thr_path.write_text(f"{threshold:.6f}\n")
    logger.info(f"threshold {threshold:.4f} written to {thr_path}")


def tune_threshold_roc(scores_normal: list[float], scores_abnormal: list[float]) -> float:
    """Given scores for known-good and known-bad samples, return threshold at max F1."""
    from sklearn.metrics import precision_recall_curve

    y_true = [0] * len(scores_normal) + [1] * len(scores_abnormal)
    y_scores = scores_normal + scores_abnormal
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best = int(np.argmax(f1[:-1]))  # last point has no corresponding threshold
    return float(thresholds[best])
