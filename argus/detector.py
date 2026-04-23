from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.models import Patchcore

from argus.config import DetectorConfig

logger = logging.getLogger(__name__)


@dataclass
class DefectResult:
    is_defect: bool
    score: float
    worst_frame_idx_cam1: int
    worst_frame_idx_cam2: int
    anomaly_map_cam1: np.ndarray | None = None
    anomaly_map_cam2: np.ndarray | None = None
    per_frame_scores_cam1: list[float] | None = None
    per_frame_scores_cam2: list[float] | None = None


class DefectDetector:
    """Wrapper around anomalib PatchCore. Runs inference on burst frames, aggregates scores."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._model: Patchcore | None = None
        self._device = "cpu"

    def load(self) -> None:
        ckpt_path = Path(self.cfg.model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {ckpt_path}. Run `argus calibrate` first."
            )
        model = Patchcore(
            backbone=self.cfg.backbone,
            layers=("layer2", "layer3"),
            pre_trained=True,
        )
        ckpt = torch.load(str(ckpt_path), map_location=self._device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._model = model
        logger.info(f"loaded PatchCore from {ckpt_path}")

        # Override threshold from calibration file if present
        thr_path = ckpt_path.with_suffix(".threshold")
        if thr_path.exists():
            try:
                self.threshold = float(thr_path.read_text().strip())
                logger.info(f"threshold override from {thr_path}: {self.threshold:.4f}")
            except ValueError:
                self.threshold = self.cfg.threshold
        else:
            self.threshold = self.cfg.threshold

    def _preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        tensors = []
        h, w = self.cfg.image_size
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (w, h))
            t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        return torch.stack(tensors).to(self._device)

    def _score_batch(self, frames: list[np.ndarray]) -> tuple[list[float], list[np.ndarray]]:
        if not frames:
            return [], []
        assert self._model is not None, "call .load() first"
        batch = self._preprocess(frames)
        with torch.no_grad():
            # PatchCore's underlying torch model accepts a raw image tensor
            out = self._model.model(batch)
        scores = out.pred_score.cpu().numpy().tolist() if out.pred_score is not None else []
        maps: list[np.ndarray] = []
        if out.anomaly_map is not None:
            am = out.anomaly_map.cpu().numpy()
            maps = [am[i] for i in range(am.shape[0])]
        return scores, maps

    def infer(
        self,
        frames_cam1: list[np.ndarray],
        frames_cam2: list[np.ndarray],
    ) -> DefectResult:
        scores1, maps1 = self._score_batch(frames_cam1)
        scores2, maps2 = self._score_batch(frames_cam2)

        best1 = int(np.argmax(scores1)) if scores1 else -1
        best2 = int(np.argmax(scores2)) if scores2 else -1
        max1 = scores1[best1] if scores1 else 0.0
        max2 = scores2[best2] if scores2 else 0.0
        score = max(max1, max2)

        threshold = getattr(self, "threshold", self.cfg.threshold)
        return DefectResult(
            is_defect=score > threshold,
            score=float(score),
            worst_frame_idx_cam1=best1,
            worst_frame_idx_cam2=best2,
            anomaly_map_cam1=maps1[best1] if maps1 and best1 >= 0 else None,
            anomaly_map_cam2=maps2[best2] if maps2 and best2 >= 0 else None,
            per_frame_scores_cam1=scores1,
            per_frame_scores_cam2=scores2,
        )
