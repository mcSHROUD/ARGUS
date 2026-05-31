"""
Synthetic defect generator for cup images.

Generates three defect types on normal frames:
  scratch  — curved lines with irregular edges
  dent     — local displacement + shadow/highlight
  spot     — irregular contamination blobs

Usage:
    python scripts/generate_defects.py \
        --input  data/raw/good/cam1 \
        --output data/raw/defects/cam1 \
        --count  200
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────
# Cup mask detection
# ──────────────────────────────────────────────────────────

def _find_cup_mask(img: np.ndarray, brightness_thresh: int = 140) -> np.ndarray:
    """Return binary mask of the cup (bright region) via thresholding + morphology."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, brightness_thresh, 255, cv2.THRESH_BINARY)
    # Remove small noise, fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Erode inward so defects don't land on the very edge
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask = cv2.erode(mask, erode_k)
    return mask


def _sample_point_in_mask(mask: np.ndarray, rng: random.Random) -> tuple[int, int] | None:
    """Return a random (x, y) pixel that is inside the cup mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    idx = rng.randint(0, len(xs) - 1)
    return int(xs[idx]), int(ys[idx])


# ──────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────

def _bezier_points(p0, p1, p2, n=60):
    """Quadratic Bezier curve."""
    t = np.linspace(0, 1, n)
    t = t[:, np.newaxis]
    pts = (
        (1 - t) ** 2 * np.array(p0)
        + 2 * (1 - t) * t * np.array(p1)
        + t ** 2 * np.array(p2)
    )
    return pts.astype(np.int32)


def _random_mask_like(img):
    return np.zeros(img.shape[:2], dtype=np.uint8)


# ──────────────────────────────────────────────────────────
# Defect generators
# ──────────────────────────────────────────────────────────

def add_scratch(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Curved scratch / cut with rough edges and natural color."""
    out = img.copy()
    h, w = out.shape[:2]

    mask = _find_cup_mask(img)
    pt = _sample_point_in_mask(mask, rng)
    if pt is None:
        # fallback: center of image
        pt = (w // 2, h // 2)
    x0, y0 = pt
    length = rng.randint(int(min(h, w) * 0.15), int(min(h, w) * 0.45))
    angle = rng.uniform(0, np.pi * 2)
    x1 = int(np.clip(x0 + length * np.cos(angle), 0, w - 1))
    y1 = int(np.clip(y0 + length * np.sin(angle), 0, h - 1))

    # Control point for curvature
    cx = int((x0 + x1) / 2 + rng.randint(-length // 3, length // 3))
    cy = int((y0 + y1) / 2 + rng.randint(-length // 3, length // 3))
    pts = _bezier_points((x0, y0), (cx, cy), (x1, y1), n=80)

    # Scratch color: darker or lighter than local surface
    local = out[max(0, y0 - 20):y0 + 20, max(0, x0 - 20):x0 + 20]
    base_brightness = int(local.mean()) if local.size else 120
    if rng.random() < 0.6:  # dark scratch (most common)
        color = tuple(max(0, base_brightness - rng.randint(70, 140)) for _ in range(3))
    else:  # bright reflection / metallic scratch
        color = tuple(min(255, base_brightness + rng.randint(80, 150)) for _ in range(3))

    thickness = rng.randint(2, 6)
    # Draw main line
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)

    # Rough edges: jitter a few short strokes along the path
    for i in range(0, len(pts), 6):
        px, py = pts[i]
        jx = px + rng.randint(-3, 3)
        jy = py + rng.randint(-3, 3)
        jx = int(np.clip(jx, 0, w - 1))
        jy = int(np.clip(jy, 0, h - 1))
        cv2.line(out, (px, py), (jx, jy), color, 1, cv2.LINE_AA)

    return out


def add_dent(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Local surface dent: pixel displacement + shadow on one side, highlight on other."""
    out = img.copy().astype(np.float32)
    h, w = out.shape[:2]

    mask = _find_cup_mask(img)
    pt = _sample_point_in_mask(mask, rng)
    if pt is None:
        pt = (w // 2, h // 2)
    cx, cy = pt
    rx = rng.randint(int(min(h, w) * 0.08), int(min(h, w) * 0.20))
    ry = int(rx * rng.uniform(0.5, 1.5))
    angle_deg = rng.randint(0, 180)

    # Build displacement map (local warp inward)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))

    # Distance from ellipse center (normalised)
    cos_a = np.cos(np.deg2rad(angle_deg))
    sin_a = np.sin(np.deg2rad(angle_deg))
    dx = map_x - cx
    dy = map_y - cy
    rx_d = (dx * cos_a + dy * sin_a) / rx
    ry_d = (-dx * sin_a + dy * cos_a) / ry
    dist = np.sqrt(rx_d ** 2 + ry_d ** 2)

    # Smooth falloff: only pixels inside 2×ellipse radius
    strength = rng.uniform(10, 25)
    falloff = np.clip(1.0 - dist / 2.0, 0, 1) ** 2
    map_x = map_x + dx * falloff * (strength / max(rx, ry))
    map_y = map_y + dy * falloff * (strength / max(rx, ry))
    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)

    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR).astype(np.float32)

    # Shadow on one side, subtle highlight on other
    light_dir = rng.uniform(0, np.pi * 2)
    shadow = (dx * np.cos(light_dir) + dy * np.sin(light_dir)) / max(rx, ry)
    shadow = np.clip(shadow * falloff, -1, 1)
    shadow_map = (shadow * rng.uniform(40, 80))[..., np.newaxis]

    result = np.clip(warped + shadow_map, 0, 255).astype(np.uint8)

    # Blend: only replace pixels inside the dent zone
    blend_mask = (falloff > 0.05).astype(np.float32)[..., np.newaxis]
    out = (result * blend_mask + img.astype(np.float32) * (1 - blend_mask)).astype(np.uint8)
    return out


def add_spot(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Irregular contamination blob: oil, grease, dust, or dark smudge."""
    out = img.copy()
    h, w = out.shape[:2]

    mask = _find_cup_mask(img)
    pt = _sample_point_in_mask(mask, rng)
    if pt is None:
        pt = (w // 2, h // 2)
    cx, cy = pt
    size = rng.randint(int(min(h, w) * 0.06), int(min(h, w) * 0.18))

    # Build irregular polygon by perturbing a circle
    n_pts = rng.randint(7, 14)
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    radii = np.array([size * rng.uniform(0.5, 1.5) for _ in range(n_pts)])
    pts = np.column_stack([
        cx + (radii * np.cos(angles)).astype(int),
        cy + (radii * np.sin(angles)).astype(int),
    ]).reshape(-1, 1, 2).astype(np.int32)

    # Spot type
    spot_type = rng.choice(["oil", "grease", "dust", "dark"])
    local = out[max(0, cy - size):cy + size, max(0, cx - size):cx + size]
    base = local.mean(axis=(0, 1)) if local.size else np.array([120.0, 120.0, 120.0])

    if spot_type == "oil":
        tint = np.array([rng.randint(-20, 20), rng.randint(-20, 20), rng.randint(-20, 20)])
        spot_color = tuple(int(np.clip(base[i] * 0.40 + tint[i], 0, 255)) for i in range(3))
        alpha = rng.uniform(0.55, 0.80)
    elif spot_type == "grease":
        spot_color = tuple(int(np.clip(base[i] * 0.60 + rng.randint(5, 25), 0, 255)) for i in [2, 1, 0])
        alpha = rng.uniform(0.60, 0.85)
    elif spot_type == "dust":
        v = int(np.clip(base.mean() * 1.25 + rng.randint(20, 60), 0, 255))
        spot_color = (v, v, v)
        alpha = rng.uniform(0.50, 0.75)
    else:  # dark smudge
        spot_color = tuple(int(np.clip(base[i] * rng.uniform(0.15, 0.35), 0, 255)) for i in range(3))
        alpha = rng.uniform(0.65, 0.90)

    # Draw filled polygon on overlay then blend
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], spot_color)

    # Gaussian blur to soften edges
    blur_k = max(3, (size // 4) | 1)  # odd kernel
    cv2.GaussianBlur(overlay, (blur_k, blur_k), 0, dst=overlay)

    out = cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)
    return out


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

DEFECT_FNS = {
    "scratch": add_scratch,
    "dent": add_dent,
    "spot": add_spot,
}


def generate(input_dir: Path, output_dir: Path, count: int, seed: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    if not sources:
        raise RuntimeError(f"No images in {input_dir}")

    rng = random.Random(seed)
    defect_types = list(DEFECT_FNS.keys())
    counts = {d: 0 for d in defect_types}

    for i in range(count):
        src = rng.choice(sources)
        img = cv2.imread(str(src))
        if img is None:
            continue

        defect = defect_types[i % len(defect_types)]  # balanced distribution
        fn = DEFECT_FNS[defect]

        # Optionally stack two defects for more variety
        result = fn(img, rng)
        if rng.random() < 0.25:
            second = rng.choice(defect_types)
            result = DEFECT_FNS[second](result, rng)

        out_path = output_dir / f"{defect}_{i:04d}.jpg"
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        counts[defect] += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{count} generated")

    print(f"Done: {count} defect images -> {output_dir}")
    for d, n in counts.items():
        print(f"  {d}: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic defect generator for cup images")
    parser.add_argument("--input", default="data/raw/good/cam1", help="Normal frames directory")
    parser.add_argument("--output", default="data/raw/defects/cam1", help="Output directory")
    parser.add_argument("--count", type=int, default=200, help="Number of defect images to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(Path(args.input), Path(args.output), args.count, args.seed)
