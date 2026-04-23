from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from argus.camera import BurstEvent
from argus.config import Config

logger = logging.getLogger(__name__)


@dataclass
class CupEvent:
    ts: datetime
    cam1_frames: list[np.ndarray]
    cam2_frames: list[np.ndarray]


class Orchestrator:
    """Matches BurstEvents from 2 cameras into CupEvent, runs detector, writes to storage."""

    def __init__(self, config: Config):
        self.config = config

    async def run(self) -> None:
        raise NotImplementedError("Implemented in Stage 4 (Pipeline + Storage integration)")
