from __future__ import annotations

from pathlib import Path

import pytest

from argus.config import load_config


def test_default_config_loads():
    cfg = load_config(Path(__file__).parent.parent / "config.yaml")
    assert len(cfg.cameras) == 2
    assert cfg.cameras[0].id == "cam1"
    assert cfg.cameras[1].id == "cam2"
    assert cfg.burst.frames == 5


def test_config_rejects_single_camera(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
cameras:
  - id: cam1
    device: /dev/video0
    roi: [0, 0, 100, 100]
motion: {}
burst: {}
sync: {}
detector:
  model_path: /tmp/m.ckpt
storage:
  db_path: /tmp/argus.db
  images_dir: /tmp/argus
ui: {}
"""
    )
    with pytest.raises(ValueError, match="exactly 2 cameras"):
        load_config(bad)
