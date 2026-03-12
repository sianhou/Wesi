from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from wesi.domain.models import Horizon, HorizonSet, Receiver, Shot


JSONDict = dict[str, Any]


PROJECT_DIRS = [
    "raw",
    "cache",
    "cache/normalized",
    "jobs",
]


def ensure_project_layout(root: Path) -> None:
    for relative in PROJECT_DIRS:
        (root / relative).mkdir(parents=True, exist_ok=True)



def write_json(path: Path, payload: JSONDict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path



def read_json(path: Path) -> JSONDict:
    return json.loads(path.read_text(encoding="utf-8"))



def save_array(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array, dtype=np.float32))
    return path



def load_array(path: Path) -> np.ndarray:
    return np.load(path)



def import_velocity_source(source: Path | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return np.asarray(source, dtype=np.float32)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(source), dtype=np.float32)
    if suffix in {".sgy", ".segy"}:
        try:
            import segyio  # type: ignore
        except ImportError as exc:
            raise RuntimeError("SEG-Y import requires the optional 'segyio' dependency") from exc
        with segyio.open(str(source), "r", ignore_geometry=True) as segy_file:
            cube = segyio.tools.cube(segy_file)
        return np.asarray(cube, dtype=np.float32)
    raise ValueError(f"Unsupported velocity source: {source}")



def import_shot_source(source: Path | list[dict[str, Any]]) -> list[Shot]:
    if isinstance(source, list):
        return [_shot_from_dict(item) for item in source]
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = read_json(source)
        return [_shot_from_dict(item) for item in payload["shots"]]
    if suffix in {".sgy", ".segy"}:
        raise RuntimeError("SEG-Y shot import requires survey sidecar support and is not enabled in the MVP scaffold")
    raise ValueError(f"Unsupported shot source: {source}")



def import_horizon_source(source: Path | HorizonSet) -> HorizonSet:
    if isinstance(source, HorizonSet):
        return source
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = read_json(source)
        return HorizonSet(horizons=[Horizon(name=item["name"], samples=[tuple(sample) for sample in item["samples"]]) for item in payload["horizons"]])
    if suffix == ".csv":
        rows: dict[str, list[tuple[int, int, float]]] = {}
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.setdefault(row["name"], []).append((int(row["x"]), int(row["y"]), float(row["z"])))
        return HorizonSet(horizons=[Horizon(name=name, samples=samples) for name, samples in rows.items()])
    raise ValueError(f"Unsupported horizon source: {source}")



def copy_raw_if_needed(root: Path, source: Path | np.ndarray | list[dict[str, Any]] | HorizonSet, target_name: str) -> Path | None:
    if isinstance(source, Path):
        target = root / "raw" / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target
    return None



def persist_shots(path: Path, shots: list[Shot]) -> Path:
    return write_json(path, {"shots": [shot.as_dict() for shot in shots]})



def load_shots(path: Path) -> list[Shot]:
    payload = read_json(path)
    return [_shot_from_dict(item) for item in payload["shots"]]



def persist_horizons(path: Path, horizons: HorizonSet) -> Path:
    return write_json(path, horizons.as_dict())



def load_horizons(path: Path) -> HorizonSet:
    return import_horizon_source(path)



def _shot_from_dict(item: JSONDict) -> Shot:
    return Shot(
        shot_id=str(item["shot_id"]),
        source=tuple(item["source"]),
        receivers=[Receiver(**receiver) for receiver in item["receivers"]],
        wavelet=[float(value) for value in item["wavelet"]],
        observed_data=item.get("observed_data"),
        metadata=item.get("metadata", {}),
    )
