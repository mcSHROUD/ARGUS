from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from argus import __version__
from argus.config import load_config


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@click.group()
@click.version_option(version=__version__, prog_name="argus")
@click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    type=click.Path(),
    help="Path to config.yaml (overrides ARGUS_CONFIG env var)",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None) -> None:
    """ARGUS — defect logging for cup production line."""
    try:
        ctx.obj = load_config(config_path)
    except FileNotFoundError as e:
        click.echo(f"error: {e}", err=True)
        ctx.exit(1)
    setup_logging(ctx.obj.logging.level)


@cli.command()
@click.pass_obj
def doctor(config) -> None:
    """Run system self-check: cameras reachable, paths writable, deps importable."""
    from argus.camera import CameraWorker

    click.echo("=== ARGUS doctor ===")
    click.echo(f"argus version: {__version__}")

    click.echo("\n[cameras]")
    for cam in config.cameras:
        click.echo(f"  {cam.id} @ {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)")
        try:
            worker = CameraWorker(cam, config.motion, config.burst)
            ok = worker.probe()
            click.echo(f"    probe: {'OK' if ok else 'FAIL (no frame)'}")
        except Exception as e:
            click.echo(f"    probe: FAIL ({e})")

    click.echo("\n[storage]")
    for p in (config.storage.db_path, config.storage.images_dir, config.detector.model_path):
        parent = Path(p).parent
        exists = parent.exists()
        writable = exists and parent.is_dir()
        click.echo(f"  {p} -> parent exists: {exists}, writable: {writable}")

    click.echo("\n[deps]")
    for mod in ("cv2", "numpy", "fastapi", "anomalib", "torch"):
        try:
            imported = __import__(mod)
            ver = getattr(imported, "__version__", "?")
            click.echo(f"  {mod}: OK ({ver})")
        except ImportError as e:
            click.echo(f"  {mod}: FAIL ({e})")

    click.echo("\nDone.")


@cli.command()
@click.pass_obj
def run(config) -> None:
    """Start the live detection pipeline (requires trained model)."""
    from argus.pipeline import Orchestrator

    orch = Orchestrator(config)
    orch.run()


@cli.command()
@click.option(
    "--normal",
    "normal_dir",
    default="/data/raw/good/cam1",
    type=click.Path(),
    help="Folder of 'good' cup frames (from extract-video or live capture)",
)
@click.option(
    "--abnormal",
    "abnormal_dir",
    default=None,
    type=click.Path(),
    help="Optional folder of defective cup frames (for validation/threshold tuning)",
)
@click.option(
    "--out",
    "out_dir",
    default="/models",
    type=click.Path(),
    help="Where to save checkpoint (default /models)",
)
@click.pass_obj
def calibrate(config, normal_dir: str, abnormal_dir: str | None, out_dir: str) -> None:
    """Train PatchCore on normal frames, tune threshold."""
    from argus import calibrate as cal

    normal = Path(normal_dir)
    abnormal = Path(abnormal_dir) if abnormal_dir else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = cal.train(config, normal, abnormal, out)
    click.echo(f"checkpoint: {ckpt}")


@cli.command()
@click.pass_obj
def ui(config) -> None:
    """Launch the review web UI."""
    import uvicorn

    uvicorn.run(
        "argus.ui.app:app",
        host=config.ui.host,
        port=config.ui.port,
        log_level=config.logging.level.lower(),
    )


@cli.command()
@click.argument("video_cam1", type=click.Path(exists=True))
@click.argument("video_cam2", type=click.Path(exists=False), required=False)
@click.pass_obj
def replay(config, video_cam1: str, video_cam2: str | None) -> None:
    """Replay one or two pre-recorded videos through the pipeline.

    Useful for end-to-end testing without real cameras attached.
    If only one video is given, the second camera is omitted.
    """
    from argus.pipeline import Orchestrator

    sources = (video_cam1, video_cam2 if video_cam2 else None)
    if video_cam2:
        click.echo(f"replay: cam1={video_cam1} cam2={video_cam2}")
    else:
        click.echo(f"replay: cam1={video_cam1} (no cam2 — events will be unpaired)")
    orch = Orchestrator(config, sources=sources)
    orch.run()


@cli.command("extract-video")
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.option("--cam", default="cam1", help="Which camera config to use (for ROI)")
@click.option(
    "--out",
    "out_dir",
    default="data/raw/good/cam1",
    type=click.Path(),
    help="Output directory for extracted burst frames",
)
@click.option(
    "--no-roi",
    is_flag=True,
    help="Extract from full frame instead of applying camera ROI",
)
@click.pass_obj
def extract_video(config, video: str, cam: str, out_dir: str, no_roi: bool) -> None:
    """Extract burst frames from a recorded video using motion-FSM."""
    from argus.extract import extract_bursts, save_bursts

    cam_cfg = next((c for c in config.cameras if c.id == cam), None)
    if cam_cfg is None:
        click.echo(f"error: camera '{cam}' not found in config", err=True)
        sys.exit(1)

    roi = None if no_roi else cam_cfg.roi
    click.echo(f"extracting from {video} | roi={roi}")

    bursts = extract_bursts(
        Path(video),
        roi=roi,
        motion_cfg=config.motion,
        burst_cfg=config.burst,
    )
    if not bursts:
        click.echo("no bursts detected — try --no-roi or tune motion.drum_stop_threshold")
        sys.exit(0)

    total = save_bursts(bursts, Path(out_dir))
    click.echo(f"saved {total} frames across {len(bursts)} bursts to {out_dir}")


@cli.command("diagnose-motion")
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.option("--cam", default="cam1", help="Camera config to use for ROI (default: cam1)")
@click.option("--no-roi", is_flag=True, help="Skip ROI crop, use full frame")
@click.option("--csv", "csv_path", default=None, type=click.Path(), help="Save frame scores to CSV")
@click.option("--plot", "plot_path", default=None, type=click.Path(), help="Save histogram PNG (requires matplotlib)")
@click.pass_obj
def diagnose_motion(
    config, video: str, cam: str, no_roi: bool, csv_path: str | None, plot_path: str | None
) -> None:
    """Analyse frame-diff scores from a recorded video.

    Prints score statistics and suggests drum_stop_threshold / drum_move_threshold
    values for config.yaml. Run on 5-10 min of production video before calibrating.

    Example:\n
        argus diagnose-motion line.mp4 --plot motion.png
    """
    import csv as csv_mod

    import cv2
    import numpy as np

    from argus.motion_fsm import frame_diff_score

    DOWNSAMPLE = 4

    cam_cfg = next((c for c in config.cameras if c.id == cam), None)
    if cam_cfg is None:
        click.echo(f"error: camera '{cam}' not found in config", err=True)
        sys.exit(1)

    roi = None if no_roi else cam_cfg.roi

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        click.echo(f"error: cannot open {video}", err=True)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    click.echo(f"video  : {video}")
    click.echo(f"frames : {total_frames}  fps: {fps:.1f}  duration: {total_frames / fps:.1f}s")
    click.echo(f"roi    : {roi or 'full frame'}")

    scores: list[float] = []
    prev_small = None

    with click.progressbar(length=total_frames, label="analysing") as bar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y : y + h, x : x + w]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(
                gray,
                (max(1, gray.shape[1] // DOWNSAMPLE), max(1, gray.shape[0] // DOWNSAMPLE)),
            )
            if prev_small is not None:
                scores.append(frame_diff_score(prev_small, small))
            prev_small = small
            bar.update(1)

    cap.release()

    if not scores:
        click.echo("error: no frames read", err=True)
        sys.exit(1)

    arr = np.array(scores)
    p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90])

    click.echo(f"\n{'─' * 42}")
    click.echo(f"  frames analysed : {len(scores)}")
    click.echo(f"  min             : {arr.min():.4f}")
    click.echo(f"  p10             : {p10:.4f}")
    click.echo(f"  p25             : {p25:.4f}")
    click.echo(f"  median (p50)    : {p50:.4f}")
    click.echo(f"  p75             : {p75:.4f}")
    click.echo(f"  p90             : {p90:.4f}")
    click.echo(f"  max             : {arr.max():.4f}")
    click.echo(f"{'─' * 42}")
    click.echo("\n[suggested values for config.yaml → motion:]")
    click.echo(f"  drum_stop_threshold: {p25:.3f}   # ≈ p25 (тихие кадры)")
    click.echo(f"  drum_move_threshold: {p75:.3f}   # ≈ p75 (кадры с движением)")
    click.echo("\n[current config.yaml values:]")
    click.echo(f"  drum_stop_threshold: {config.motion.drum_stop_threshold}")
    click.echo(f"  drum_move_threshold: {config.motion.drum_move_threshold}")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.writer(f)
            writer.writerow(["frame", "diff_score"])
            for i, s in enumerate(scores):
                writer.writerow([i, f"{s:.6f}"])
        click.echo(f"\nCSV → {csv_path}")

    if plot_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(arr, bins=100, color="#4488cc", alpha=0.7, edgecolor="none")
            ax.axvline(p25, color="#00aa00", linestyle="--", linewidth=1.5,
                       label=f"stop_threshold = {p25:.3f} (p25)")
            ax.axvline(p75, color="#cc0000", linestyle="--", linewidth=1.5,
                       label=f"move_threshold = {p75:.3f} (p75)")
            ax.set_xlabel("frame_diff_score")
            ax.set_ylabel("кадров")
            ax.set_title(f"Распределение motion-score — {Path(video).name}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            click.echo(f"plot  → {plot_path}")
        except ImportError:
            click.echo("matplotlib не установлен — plot пропущен")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
