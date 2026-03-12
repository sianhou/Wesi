from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


JSONDict = dict[str, Any]


@dataclass(slots=True)
class GridSpec:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def as_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class OffsetRule:
    inline_padding: int
    crossline_padding: int
    depth_padding: int
    min_offset: float = 0.0
    max_offset: float | None = None
    pml: int = 10

    def as_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class Receiver:
    x: int
    y: int
    z: int = 0

    def as_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class Shot:
    shot_id: str
    source: tuple[int, int, int]
    receivers: list[Receiver]
    wavelet: list[float]
    observed_data: list[list[float]] | None = None
    metadata: JSONDict = field(default_factory=dict)

    def as_dict(self) -> JSONDict:
        payload = asdict(self)
        payload["receivers"] = [receiver.as_dict() for receiver in self.receivers]
        return payload


@dataclass(slots=True)
class Horizon:
    name: str
    samples: list[tuple[int, int, float]]

    def as_dict(self) -> JSONDict:
        return {"name": self.name, "samples": [list(sample) for sample in self.samples]}


@dataclass(slots=True)
class HorizonSet:
    horizons: list[Horizon]

    def as_dict(self) -> JSONDict:
        return {"horizons": [horizon.as_dict() for horizon in self.horizons]}


@dataclass(slots=True)
class CheckpointPolicy:
    save_forward_wavefield: bool = True
    checkpoint_stride: int = 1

    def as_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(slots=True)
class ProjectConfig:
    project_id: str
    name: str
    root: Path
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_threads: int = 1
    fd_order: int = 2
    coordinate_system: str = "local-cartesian"
    metadata: JSONDict = field(default_factory=dict)

    def as_dict(self) -> JSONDict:
        payload = asdict(self)
        payload["root"] = str(self.root)
        return payload


@dataclass(slots=True)
class Submodel:
    submodel_id: str
    velocity_id: str
    shot_id: str
    bounds: tuple[int, int, int, int, int, int]
    halo: int
    pml: int
    velocity_path: Path
    grid: GridSpec
    horizon_path: Path | None = None
    metadata: JSONDict = field(default_factory=dict)

    def as_dict(self) -> JSONDict:
        payload = asdict(self)
        payload["velocity_path"] = str(self.velocity_path)
        payload["horizon_path"] = str(self.horizon_path) if self.horizon_path else None
        payload["grid"] = self.grid.as_dict()
        return payload


@dataclass(slots=True)
class Subtask:
    subtask_id: str
    project_id: str
    shot_id: str
    submodel_id: str
    run_params: JSONDict
    checkpoint_policy: CheckpointPolicy
    status: str = "READY"
    stage: str = "full"

    def as_dict(self) -> JSONDict:
        payload = asdict(self)
        payload["checkpoint_policy"] = self.checkpoint_policy.as_dict()
        return payload


@dataclass(slots=True)
class JobConfig:
    stages: tuple[str, ...] = ("forward", "rtm")
    subtask_ids: list[str] = field(default_factory=list)
    max_workers: int = 1
    threads_per_worker: int = 1
    allow_resume: bool = True

    def as_dict(self) -> JSONDict:
        payload = asdict(self)
        payload["stages"] = list(self.stages)
        return payload


@dataclass(slots=True)
class Job:
    job_id: str
    project_id: str
    config: JobConfig
    status: str = "CREATED"
    metadata: JSONDict = field(default_factory=dict)

    def as_dict(self) -> JSONDict:
        payload = asdict(self)
        payload["config"] = self.config.as_dict()
        return payload


@dataclass(slots=True)
class JobArtifact:
    job_id: str
    subtask_id: str
    artifact_type: str
    path: Path
    metadata: JSONDict = field(default_factory=dict)

    def as_dict(self) -> JSONDict:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload
