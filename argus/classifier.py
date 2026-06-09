from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

logger = logging.getLogger(__name__)

IMAGE_SIZE = 256

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class _DefectHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = efficientnet_b0(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)


class DefectClassifier:
    """EfficientNet-B0 binary classifier: normal (0) vs defect (1)."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.threshold: float = 0.5
        self._model: _DefectHead | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Classifier not found: {self.model_path}. "
                "Run: python scripts/train_classifier.py"
            )
        ckpt = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
        model = _DefectHead()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._model = model
        thr_path = self.model_path.with_suffix(".threshold")
        if thr_path.exists():
            self.threshold = float(thr_path.read_text(encoding="utf-8-sig").strip())
        else:
            self.threshold = float(ckpt.get("threshold", 0.5))
        logger.info(f"classifier loaded from {self.model_path}  threshold={self.threshold:.4f}")

    def predict(self, img: np.ndarray) -> tuple[float, bool]:
        """Return (probability 0-1, is_defect)."""
        assert self._model is not None, "call .load() first"
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = _TRANSFORM(rgb).unsqueeze(0)
        with torch.no_grad():
            prob = float(torch.sigmoid(self._model(t)).item())
        return prob, prob > self.threshold

    def predict_with_gradcam(self, img: np.ndarray) -> tuple[float, bool, np.ndarray | None]:
        """Return (probability 0-1, is_defect, gradcam_map or None).

        gradcam_map is a float32 array (H, W) with raw activation weights —
        same format as PatchCore anomaly_map, ready for _prepare_images().
        """
        assert self._model is not None, "call .load() first"
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = _TRANSFORM(rgb).unsqueeze(0)

        fmaps: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []

        # Hook the last conv block of the backbone (features[-1])
        target_layer = self._model.backbone[0][-1]
        fwd = target_layer.register_forward_hook(lambda m, i, o: fmaps.append(o))
        bwd = target_layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))

        self._model.eval()
        logit = self._model(t)
        prob = float(torch.sigmoid(logit).item())

        if prob > self.threshold:
            self._model.zero_grad()
            logit.backward()

        fwd.remove()
        bwd.remove()

        if prob <= self.threshold or not grads:
            return prob, False, None

        # Grad-CAM: weight each channel by its mean gradient
        g = grads[0].squeeze(0)   # (C, H, W)
        f = fmaps[0].squeeze(0)   # (C, H, W)
        weights = g.mean(dim=(1, 2))          # (C,)
        cam = (weights[:, None, None] * f).sum(dim=0)  # (H, W)
        cam = torch.relu(cam).detach().numpy().astype(np.float32)

        return prob, True, cam
