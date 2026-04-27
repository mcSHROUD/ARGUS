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


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
