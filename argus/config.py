from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class CameraConfig(BaseModel):
    id: str
    device: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    roi: tuple[int, int, int, int] = Field(description="x, y, w, h")


class MotionConfig(BaseModel):
    history: int = 500  # reserved for MOG2-based worker (live intrusion detection)
    var_threshold: int = 16
    drum_stop_threshold: float = 0.045  # frame-diff: <= this = drum is stopped
    drum_move_threshold: float = 0.08  # frame-diff: >= this = drum is rotating
    stop_window_ms: int = 150


class BurstConfig(BaseModel):
    frames: int = 5
    max_duration_ms: int = 800
    min_sharpness: float = 30.0


class SyncConfig(BaseModel):
    window_ms: int = 300


class DetectorConfig(BaseModel):
    model_path: Path
    backbone: str = "wide_resnet50_2"
    threshold: float = 0.5
    image_size: tuple[int, int] = (256, 256)


class StorageConfig(BaseModel):
    db_path: Path
    images_dir: Path
    save_worst_only: bool = True


class UIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseModel):
    cameras: list[CameraConfig]
    motion: MotionConfig
    burst: BurstConfig
    sync: SyncConfig
    detector: DetectorConfig
    storage: StorageConfig
    ui: UIConfig
    logging: LoggingConfig = LoggingConfig()

    @field_validator("cameras")
    @classmethod
    def exactly_two_cameras(cls, v: list[CameraConfig]) -> list[CameraConfig]:
        if len(v) != 2:
            raise ValueError(f"ARGUS requires exactly 2 cameras, got {len(v)}")
        ids = [c.id for c in v]
        if len(set(ids)) != 2:
            raise ValueError(f"Camera ids must be unique, got {ids}")
        return v


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = os.environ.get("ARGUS_CONFIG", "config.yaml")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
