#!/usr/bin/env python3
"""Train EfficientNet-B0 binary defect classifier.

Strategy: frozen backbone (ImageNet features) + small MLP head.
With ~30 defect examples augmentation expands them to ~450 — enough for a head.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --epochs 150 --augment 20
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ROOT = Path(__file__).parent.parent
NORMAL_DIR   = ROOT / "data" / "raw" / "good" / "cam1"
DEFECTS_DIR  = ROOT / "data" / "raw" / "defects_real"
OUT_DIR      = ROOT / "models_new"
IMAGE_SIZE   = 256
SEED         = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ── Transforms ────────────────────────────────────────────────────────────────

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225],
)

DEFECT_AUG = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=40),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), scale=(0.85, 1.15)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    NORMALIZE,
])

NORMAL_AUG = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    NORMALIZE,
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    NORMALIZE,
])


# ── Dataset ───────────────────────────────────────────────────────────────────

def preload_images(files: list[Path]) -> dict[Path, np.ndarray]:
    """Read all unique images once and keep them in RAM as RGB arrays."""
    cache: dict[Path, np.ndarray] = {}
    for f in set(files):
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        cache[f] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cache


class CupDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        labels: list[int],
        transform,
        cache: dict[Path, np.ndarray],
        augment_defects: int = 1,
    ):
        self._cache = cache
        self.samples: list[tuple[Path, int, transforms.Compose]] = []
        for f, lab in zip(files, labels):
            repeat = augment_defects if lab == 1 else 1
            for _ in range(repeat):
                self.samples.append((f, lab, transform))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, tfm = self.samples[idx]
        pil = Image.fromarray(self._cache[path])
        return tfm(pil), torch.tensor(label, dtype=torch.float32)


# ── Model ─────────────────────────────────────────────────────────────────────

class DefectHead(nn.Module):
    """EfficientNet-B0 with last 4 feature blocks unfrozen for fine-tuning.

    EfficientNet-B0 features layout (indices 0-8):
      0: stem conv  1-8: MBConv stages
    We freeze 0-4, unfreeze 5-8 (last 4 stages) so the network can learn
    cup-specific texture patterns instead of relying only on ImageNet features.
    """

    def __init__(self):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Freeze all first
        for p in base.parameters():
            p.requires_grad = False
        # Unfreeze last 4 MBConv stages + top conv (indices 5-8 of features)
        features = base.features
        for block in list(features.children())[-4:]:
            for p in block.parameters():
                p.requires_grad = True

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)

    def param_groups(self, lr_head: float, lr_backbone: float) -> list[dict]:
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params     = list(self.head.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ]


# ── Training ──────────────────────────────────────────────────────────────────

def collect_files() -> tuple[list[Path], list[int]]:
    normals = sorted(NORMAL_DIR.glob("*.jpg")) + sorted(NORMAL_DIR.glob("*.png"))
    defects: list[Path] = []
    for sub in sorted(DEFECTS_DIR.iterdir()):
        if sub.is_dir():
            defects += sorted(sub.glob("*.jpg")) + sorted(sub.glob("*.png"))

    if not normals:
        raise RuntimeError(f"No normal images found in {NORMAL_DIR}")
    if not defects:
        raise RuntimeError(f"No real defect images found in {DEFECTS_DIR}")

    print(f"  Normal : {len(normals)} images")
    print(f"  Defects: {len(defects)} images")

    files  = normals + defects
    labels = [0] * len(normals) + [1] * len(defects)
    return files, labels


def split(files, labels, val_ratio_normal=0.2):
    """All defects go to training. Normals split 80/20 to monitor FPR."""
    rng = random.Random(SEED)
    train_f, train_l, val_f, val_l = [], [], [], []
    for cls in [0, 1]:
        cls_files = [f for f, l in zip(files, labels) if l == cls]
        rng.shuffle(cls_files)
        if cls == 1:
            # All defects in training — too few to spare any for val
            train_f += cls_files
            train_l += [1] * len(cls_files)
        else:
            n_val = max(1, int(len(cls_files) * val_ratio_normal))
            val_f  += cls_files[:n_val];  val_l  += [0] * n_val
            train_f += cls_files[n_val:]; train_l += [0] * (len(cls_files) - n_val)
    return train_f, train_l, val_f, val_l


def find_threshold(model: DefectHead, val_loader: DataLoader, device: str) -> float:
    """F1-optimal threshold on validation set."""
    from sklearn.metrics import precision_recall_curve
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device))
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            trues.extend(y.tolist())
    precision, recall, thresholds = precision_recall_curve(trues, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best])


def train(epochs: int = 100, augment: int = 15, lr: float = 1e-3) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print("\nCollecting images...")
    files, labels = collect_files()

    train_f, train_l, val_f, val_l = split(files, labels)
    n_def_train = sum(train_l)
    print(f"\nTrain: {len(train_f)} total  ({len(train_f)-n_def_train} normal, {n_def_train} defect x{augment} aug = {n_def_train*augment} effective)")
    print(f"Val  : {len(val_f)} total  ({sum(1 for l in val_l if l==0)} normal, {sum(val_l)} defect)")

    # Load all images ONCE into RAM — shared across all dataset splits
    print("  loading images to RAM...", flush=True)
    cache = preload_images(files)
    print(f"  {len(cache)} images loaded ({sum(p.stat().st_size for p in cache)//1024//1024} MB)", flush=True)

    from torch.utils.data import ConcatDataset
    normal_files = [f for f, l in zip(train_f, train_l) if l == 0]
    defect_files = [f for f, l in zip(train_f, train_l) if l == 1]
    normal_ds = CupDataset(normal_files, [0]*len(normal_files), NORMAL_AUG, cache)
    defect_ds = CupDataset(defect_files, [1]*len(defect_files), DEFECT_AUG, cache, augment_defects=augment)
    train_combined = ConcatDataset([normal_ds, defect_ds])

    val_ds = CupDataset(val_f, val_l, VAL_TRANSFORM, cache)
    train_dl = DataLoader(train_combined, batch_size=64, shuffle=True,  num_workers=0, pin_memory=(device=="cuda"))
    val_dl   = DataLoader(val_ds,         batch_size=64, shuffle=False, num_workers=0, pin_memory=(device=="cuda"))

    model = DefectHead().to(device)
    n_def   = sum(train_l)
    n_norm  = len(train_l) - n_def
    pos_w   = torch.tensor([n_norm / max(n_def, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    # Two learning rates: backbone fine-tunes slowly, head learns fast
    optimizer = torch.optim.AdamW(
        model.param_groups(lr_head=lr, lr_backbone=lr * 0.05),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\nTraining for {epochs} epochs  (pos_weight={pos_w.item():.1f})\n")
    best_f1, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # ── validate every 10 epochs ──
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for x, y in val_dl:
                    logits = model(x.to(device))
                    prob = torch.sigmoid(logits).cpu()
                    preds.extend((prob > 0.5).int().tolist())
                    trues.extend(y.int().tolist())
            from sklearn.metrics import f1_score
            f1 = f1_score(trues, preds, zero_division=0)
            recall = sum(p == 1 and t == 1 for p, t in zip(preds, trues)) / max(sum(trues), 1)
            fp     = sum(p == 1 and t == 0 for p, t in zip(preds, trues))
            print(f"  Epoch {epoch:3d} | loss={total_loss/len(train_dl):.4f} | F1={f1:.3f} | recall={recall:.0%} | FP={fp}")
            if f1 >= best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── restore best & find optimal threshold ──
    if best_state:
        model.load_state_dict(best_state)

    threshold = find_threshold(model, val_dl, device)
    print(f"\nOptimal threshold (F1 on val): {threshold:.4f}")

    # ── final eval with optimal threshold ──
    model.eval()
    probs_all, trues_all = [], []
    with torch.no_grad():
        for x, y in val_dl:
            logits = model(x.to(device))
            probs_all.extend(torch.sigmoid(logits).cpu().tolist())
            trues_all.extend(y.int().tolist())
    preds_final = [int(p > threshold) for p in probs_all]
    print("\nValidation results at optimal threshold:")
    print(classification_report(trues_all, preds_final, target_names=["normal", "defect"], digits=3))
    cm = confusion_matrix(trues_all, preds_final)
    print(f"Confusion matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

    # ── save ──
    ckpt_path = OUT_DIR / "classifier.pt"
    torch.save({"state_dict": model.state_dict(), "threshold": threshold}, str(ckpt_path))
    (OUT_DIR / "classifier.threshold").write_text(f"{threshold:.6f}\n")
    print(f"\nSaved: {ckpt_path}")
    print(f"Saved: {OUT_DIR / 'classifier.threshold'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,   default=100, help="training epochs (default 100)")
    parser.add_argument("--augment", type=int,   default=15,  help="augmentation multiplier for defects (default 15)")
    parser.add_argument("--lr",      type=float, default=1e-3, help="learning rate (default 1e-3)")
    args = parser.parse_args()
    train(epochs=args.epochs, augment=args.augment, lr=args.lr)
