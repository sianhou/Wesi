from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from wesi.bindings import backend_from_name
from wesi.infrastructure.io import load_array, save_array, write_json


@dataclass(slots=True)
class ExecutionRequest:
    job_id: str
    subtask_id: str
    stages: tuple[str, ...]
    backend_name: str
    library_path: str | None
    velocity_path: str
    source: tuple[int, int, int]
    receivers: list[tuple[int, int, int]]
    wavelet: list[float]
    observed_data: list[list[float]] | None
    dt: float
    dx: float
    dy: float
    dz: float
    save_forward_wavefield: bool
    output_dir: str
    reuse_forward: bool = True


class LocalExecutor:
    def run(self, requests: Iterable[ExecutionRequest], max_workers: int = 1) -> list[dict[str, Any]]:
        request_list = list(requests)
        if max_workers <= 1:
            return [_execute_request(request) for request in request_list]
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as pool:
            return list(pool.map(_execute_request, request_list))


class MpiExecutor:
    def run(self, requests: Iterable[ExecutionRequest], max_workers: int = 1) -> list[dict[str, Any]]:
        raise NotImplementedError("MPI execution is reserved for a later version")



def _execute_request(request: ExecutionRequest) -> dict[str, Any]:
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = backend_from_name(request.backend_name, request.library_path)
    velocity = load_array(Path(request.velocity_path))
    wavelet = np.asarray(request.wavelet, dtype=np.float32)
    observed_data = np.asarray(request.observed_data, dtype=np.float32) if request.observed_data is not None else None
    checkpoints: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    log_lines: list[str] = []
    forward_wavefield = None
    recorded_data = None

    try:
        if "forward" in request.stages:
            forward_result = backend.run_forward(
                velocity=velocity,
                source=request.source,
                receivers=request.receivers,
                wavelet=wavelet,
                save_wavefield=request.save_forward_wavefield,
                dt=request.dt,
                dx=request.dx,
                dy=request.dy,
                dz=request.dz,
            )
            recorded_data = forward_result.recorded_data
            forward_wavefield = forward_result.forward_wavefield
            if recorded_data is not None:
                recorded_path = output_dir / "recorded.npy"
                save_array(recorded_path, recorded_data)
                artifacts.append({"artifact_type": "recorded", "path": str(recorded_path), "metadata": forward_result.diagnostics or {}})
            if forward_wavefield is not None:
                wavefield_path = output_dir / "forward_wavefield.npy"
                save_array(wavefield_path, forward_wavefield)
                checkpoints.append({"stage": "forward", "path": str(wavefield_path), "metadata": {"shape": list(forward_wavefield.shape)}})
                artifacts.append({"artifact_type": "forward_wavefield", "path": str(wavefield_path), "metadata": forward_result.diagnostics or {}})
            log_lines.append("forward completed")
        elif request.reuse_forward:
            wavefield_path = output_dir / "forward_wavefield.npy"
            recorded_path = output_dir / "recorded.npy"
            if wavefield_path.exists():
                forward_wavefield = load_array(wavefield_path)
            if recorded_path.exists():
                recorded_data = load_array(recorded_path)

        if "rtm" in request.stages:
            if observed_data is None:
                if recorded_data is None:
                    recorded_path = output_dir / "recorded.npy"
                    if recorded_path.exists():
                        recorded_data = load_array(recorded_path)
                if recorded_data is None:
                    raise RuntimeError("RTM requires observed data or a forward pass result")
                observed_data = recorded_data
            if forward_wavefield is None and request.reuse_forward:
                wavefield_path = output_dir / "forward_wavefield.npy"
                if wavefield_path.exists():
                    forward_wavefield = load_array(wavefield_path)
            rtm_result = backend.run_rtm(
                velocity=velocity,
                source=request.source,
                receivers=request.receivers,
                wavelet=wavelet,
                observed_data=observed_data,
                forward_wavefield=forward_wavefield,
                dt=request.dt,
                dx=request.dx,
                dy=request.dy,
                dz=request.dz,
            )
            if rtm_result.image is not None:
                image_path = output_dir / "image.npy"
                save_array(image_path, rtm_result.image)
                artifacts.append({"artifact_type": "image", "path": str(image_path), "metadata": rtm_result.diagnostics or {}})
            log_lines.append("rtm completed")
        write_json(output_dir / "manifest.json", {"status": "completed", "steps": log_lines})
        return {"subtask_id": request.subtask_id, "status": "COMPLETED", "artifacts": artifacts, "checkpoints": checkpoints, "log": log_lines}
    except Exception as exc:  # pragma: no cover - error path still persisted for ops use
        write_json(output_dir / "manifest.json", {"status": "failed", "error": str(exc), "steps": log_lines})
        return {"subtask_id": request.subtask_id, "status": "FAILED", "artifacts": artifacts, "checkpoints": checkpoints, "error": str(exc), "log": log_lines}
