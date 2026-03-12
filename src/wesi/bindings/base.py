from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class SimulationResult:
    recorded_data: np.ndarray | None = None
    forward_wavefield: np.ndarray | None = None
    image: np.ndarray | None = None
    diagnostics: dict[str, Any] | None = None


class SimulationBackend(ABC):
    name: str

    @abstractmethod
    def run_forward(
        self,
        velocity: np.ndarray,
        source: tuple[int, int, int],
        receivers: list[tuple[int, int, int]],
        wavelet: np.ndarray,
        save_wavefield: bool,
        dt: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> SimulationResult:
        raise NotImplementedError

    @abstractmethod
    def run_rtm(
        self,
        velocity: np.ndarray,
        source: tuple[int, int, int],
        receivers: list[tuple[int, int, int]],
        wavelet: np.ndarray,
        observed_data: np.ndarray,
        forward_wavefield: np.ndarray | None,
        dt: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> SimulationResult:
        raise NotImplementedError
