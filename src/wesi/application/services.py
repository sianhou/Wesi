from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from wesi.application.executors import ExecutionRequest, LocalExecutor
from wesi.domain.models import (
    CheckpointPolicy,
    GridSpec,
    Horizon,
    HorizonSet,
    Job,
    JobArtifact,
    JobConfig,
    OffsetRule,
    ProjectConfig,
    Receiver,
    Shot,
    Submodel,
    Subtask,
)
from wesi.infrastructure.io import (
    copy_raw_if_needed,
    ensure_project_layout,
    import_horizon_source,
    import_shot_source,
    import_velocity_source,
    load_array,
    load_horizons,
    load_shots,
    persist_horizons,
    persist_shots,
    save_array,
)
from wesi.infrastructure.jobstore import JobStore


class WesiService:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        ensure_project_layout(self.root)
        self.store = JobStore(self.root)
        self.store.initialize()

    def create_project(
        self,
        name: str,
        spacing: tuple[float, float, float] = (10.0, 10.0, 10.0),
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        project_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProjectConfig:
        config = ProjectConfig(
            project_id=project_id or f"project-{uuid.uuid4().hex[:8]}",
            name=name,
            root=self.root,
            spacing=spacing,
            origin=origin,
            metadata=metadata or {},
        )
        self.store.save_project(config)
        return config

    def import_velocity(self, source: str | Path | np.ndarray, name: str = "velocity") -> str:
        dataset_id = f"vel-{uuid.uuid4().hex[:8]}"
        source_obj = Path(source) if isinstance(source, (str, Path)) else source
        array = import_velocity_source(source_obj)
        if array.ndim != 3:
            raise ValueError("Velocity model must be a 3D array ordered as z, y, x")
        normalized_path = self.root / "cache" / "normalized" / f"{dataset_id}.npy"
        save_array(normalized_path, array)
        raw_path = copy_raw_if_needed(
            self.root,
            source_obj,
            f"{dataset_id}{Path(source).suffix}" if isinstance(source, (str, Path)) else f"{dataset_id}.npy",
        )
        metadata = {"name": name, "shape": list(array.shape), "dtype": str(array.dtype)}
        self.store.save_dataset(dataset_id, "velocity", normalized_path, raw_path, metadata)
        return dataset_id

    def import_shots(self, source: str | Path | list[dict[str, Any]], name: str = "survey") -> str:
        dataset_id = f"shots-{uuid.uuid4().hex[:8]}"
        source_obj = Path(source) if isinstance(source, (str, Path)) else source
        shots = import_shot_source(source_obj)
        normalized_path = self.root / "cache" / "normalized" / f"{dataset_id}.json"
        persist_shots(normalized_path, shots)
        raw_path = copy_raw_if_needed(
            self.root,
            source_obj,
            f"{dataset_id}{Path(source).suffix}" if isinstance(source, (str, Path)) else f"{dataset_id}.json",
        )
        metadata = {"name": name, "shot_count": len(shots)}
        self.store.save_dataset(dataset_id, "shots", normalized_path, raw_path, metadata)
        return dataset_id

    def import_horizons(self, source: str | Path | HorizonSet, name: str = "horizons") -> str:
        dataset_id = f"hor-{uuid.uuid4().hex[:8]}"
        source_obj = Path(source) if isinstance(source, (str, Path)) else source
        horizons = import_horizon_source(source_obj)
        normalized_path = self.root / "cache" / "normalized" / f"{dataset_id}.json"
        persist_horizons(normalized_path, horizons)
        raw_path = copy_raw_if_needed(
            self.root,
            source_obj,
            f"{dataset_id}{Path(source).suffix}" if isinstance(source, (str, Path)) else f"{dataset_id}.json",
        )
        metadata = {"name": name, "horizon_count": len(horizons.horizons)}
        self.store.save_dataset(dataset_id, "horizons", normalized_path, raw_path, metadata)
        return dataset_id

    def build_grid(
        self,
        velocity_id: str | None = None,
        spacing: tuple[float, float, float] | None = None,
        origin: tuple[float, float, float] | None = None,
    ) -> GridSpec:
        velocity_row = self.store.get_dataset(velocity_id) if velocity_id else self.store.latest_dataset("velocity")
        velocity = load_array(Path(velocity_row["path"]))
        project = self.store.get_project()
        spacing_values = spacing or tuple(float(value) for value in self._decode_json(project["spacing_json"]))
        origin_values = origin or tuple(float(value) for value in self._decode_json(project["origin_json"]))
        grid = GridSpec(
            nx=int(velocity.shape[2]),
            ny=int(velocity.shape[1]),
            nz=int(velocity.shape[0]),
            dx=float(spacing_values[0]),
            dy=float(spacing_values[1]),
            dz=float(spacing_values[2]),
            origin=origin_values,
        )
        self.store.save_grid(f"grid-{uuid.uuid4().hex[:8]}", velocity_row["dataset_id"], grid)
        return grid

    def build_subtasks(
        self,
        velocity_id: str | None = None,
        shot_dataset_id: str | None = None,
        horizon_id: str | None = None,
        offset_rule: OffsetRule | None = None,
        dt: float = 0.001,
    ) -> list[str]:
        project = self.store.get_project()
        velocity_row = self.store.get_dataset(velocity_id) if velocity_id else self.store.latest_dataset("velocity")
        shots_row = self.store.get_dataset(shot_dataset_id) if shot_dataset_id else self.store.latest_dataset("shots")
        horizon_row = self.store.get_dataset(horizon_id) if horizon_id else self._try_latest_dataset("horizons")
        velocity = load_array(Path(velocity_row["path"]))
        _, grid = self.store.latest_grid()
        shots = load_shots(Path(shots_row["path"]))
        horizons = load_horizons(Path(horizon_row["path"])) if horizon_row else HorizonSet(horizons=[])
        rule = offset_rule or OffsetRule(inline_padding=8, crossline_padding=8, depth_padding=8, pml=8)
        subtask_ids: list[str] = []

        for shot in shots:
            filtered_receivers, receiver_indices = self._filter_receivers(shot, rule)
            if not filtered_receivers:
                continue
            xs = [shot.source[0], *[receiver.x for receiver in filtered_receivers]]
            ys = [shot.source[1], *[receiver.y for receiver in filtered_receivers]]
            zs = [shot.source[2], *[receiver.z for receiver in filtered_receivers]]
            x0 = max(0, min(xs) - rule.inline_padding - rule.pml)
            x1 = min(grid.nx, max(xs) + rule.inline_padding + rule.pml + 1)
            y0 = max(0, min(ys) - rule.crossline_padding - rule.pml)
            y1 = min(grid.ny, max(ys) + rule.crossline_padding + rule.pml + 1)
            z0 = max(0, min(zs) - rule.depth_padding - rule.pml)
            z1 = min(grid.nz, max(zs) + rule.depth_padding + rule.pml + 1)
            if x0 >= x1 or y0 >= y1 or z0 >= z1:
                raise RuntimeError(f"Invalid submodel bounds for shot {shot.shot_id}")

            local_velocity = velocity[z0:z1, y0:y1, x0:x1]
            submodel_id = f"submodel-{shot.shot_id}-{uuid.uuid4().hex[:6]}"
            velocity_path = self.root / "cache" / "normalized" / "submodels" / f"{submodel_id}.npy"
            save_array(velocity_path, local_velocity)
            local_grid = GridSpec(
                nx=x1 - x0,
                ny=y1 - y0,
                nz=z1 - z0,
                dx=grid.dx,
                dy=grid.dy,
                dz=grid.dz,
                origin=(grid.origin[0] + x0 * grid.dx, grid.origin[1] + y0 * grid.dy, grid.origin[2] + z0 * grid.dz),
            )
            local_horizons = HorizonSet(horizons=self._clip_horizons(horizons.horizons, x0, x1, y0, y1, z0, z1))
            horizon_path = None
            if local_horizons.horizons:
                horizon_path = self.root / "cache" / "normalized" / "submodels" / f"{submodel_id}_horizons.json"
                persist_horizons(horizon_path, local_horizons)

            submodel = Submodel(
                submodel_id=submodel_id,
                velocity_id=velocity_row["dataset_id"],
                shot_id=shot.shot_id,
                bounds=(x0, x1, y0, y1, z0, z1),
                halo=rule.inline_padding,
                pml=rule.pml,
                velocity_path=velocity_path,
                grid=local_grid,
                horizon_path=horizon_path,
                metadata={"shot_dataset_id": shots_row["dataset_id"]},
            )
            self.store.save_submodel(submodel)

            local_source = (shot.source[0] - x0, shot.source[1] - y0, shot.source[2] - z0)
            local_receivers = [(receiver.x - x0, receiver.y - y0, receiver.z - z0) for receiver in filtered_receivers]
            observed = None
            if shot.observed_data is not None:
                observed = [[float(row[index]) for index in receiver_indices] for row in shot.observed_data]
            subtask = Subtask(
                subtask_id=f"subtask-{shot.shot_id}-{uuid.uuid4().hex[:6]}",
                project_id=project["project_id"],
                shot_id=shot.shot_id,
                submodel_id=submodel_id,
                run_params={
                    "dt": dt,
                    "local_source": list(local_source),
                    "local_receivers": [list(item) for item in local_receivers],
                    "wavelet": list(map(float, shot.wavelet)),
                    "observed_data": observed,
                    "dx": local_grid.dx,
                    "dy": local_grid.dy,
                    "dz": local_grid.dz,
                },
                checkpoint_policy=CheckpointPolicy(save_forward_wavefield=True, checkpoint_stride=1),
            )
            self.store.save_subtask(subtask)
            subtask_ids.append(subtask.subtask_id)
        return subtask_ids

    def create_job(
        self,
        subtask_ids: list[str] | None = None,
        stages: tuple[str, ...] = ("forward", "rtm"),
        max_workers: int = 1,
        threads_per_worker: int = 1,
        allow_resume: bool = True,
    ) -> str:
        project = self.store.get_project()
        selected_subtasks = subtask_ids or [subtask.subtask_id for subtask in self.store.list_subtasks()]
        job = Job(
            job_id=f"job-{uuid.uuid4().hex[:8]}",
            project_id=project["project_id"],
            config=JobConfig(
                stages=stages,
                subtask_ids=selected_subtasks,
                max_workers=max_workers,
                threads_per_worker=threads_per_worker,
                allow_resume=allow_resume,
            ),
            status="READY",
        )
        self.store.create_job(job)
        for subtask_id in selected_subtasks:
            self.store.bind_job_subtask(job.job_id, subtask_id, status="READY")
        return job.job_id

    def run_job(self, job_id: str, backend: str = "auto", library_path: str | None = None) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        self.store.update_job_status(job_id, "RUNNING")
        requests = list(
            self._build_requests(
                job_id,
                job.config.subtask_ids,
                job.config.stages,
                backend,
                library_path,
                job.config.allow_resume,
            )
        )
        executor = LocalExecutor()
        results = executor.run(requests, max_workers=job.config.max_workers)
        has_failure = False
        for result in results:
            self.store.update_subtask_status(result["subtask_id"], result["status"])
            self.store.update_job_subtask(job_id, result["subtask_id"], result["status"], result.get("error"))
            for checkpoint in result["checkpoints"]:
                self.store.register_checkpoint(
                    job_id,
                    result["subtask_id"],
                    checkpoint["stage"],
                    Path(checkpoint["path"]),
                    checkpoint["metadata"],
                )
            for artifact in result["artifacts"]:
                self.store.register_artifact(
                    JobArtifact(
                        job_id=job_id,
                        subtask_id=result["subtask_id"],
                        artifact_type=artifact["artifact_type"],
                        path=Path(artifact["path"]),
                        metadata=artifact["metadata"],
                    )
                )
            has_failure = has_failure or result["status"] != "COMPLETED"
        self.store.update_job_status(job_id, "FAILED" if has_failure else "COMPLETED")
        return {"job_id": job_id, "status": "FAILED" if has_failure else "COMPLETED", "results": results}

    def resume_job(self, job_id: str, backend: str = "auto", library_path: str | None = None) -> dict[str, Any]:
        return self.run_job(job_id, backend=backend, library_path=library_path)

    def run_partial(
        self,
        subtask_ids: list[str],
        stages: tuple[str, ...] = ("forward",),
        backend: str = "auto",
        max_workers: int = 1,
        threads_per_worker: int = 1,
        library_path: str | None = None,
    ) -> dict[str, Any]:
        job_id = self.create_job(
            subtask_ids=subtask_ids,
            stages=stages,
            max_workers=max_workers,
            threads_per_worker=threads_per_worker,
            allow_resume=True,
        )
        return self.run_job(job_id, backend=backend, library_path=library_path)

    def load_result_as_tensor(self, job_id: str, subtask_id: str, artifact_type: str = "image") -> Any:
        artifact_row = next(
            (row for row in self.store.list_artifacts(job_id, subtask_id) if row["artifact_type"] == artifact_type),
            None,
        )
        if artifact_row is None:
            raise RuntimeError(f"Artifact not found: {artifact_type} for {subtask_id}")
        array = load_array(Path(artifact_row["path"]))
        try:
            import torch
        except ImportError:
            return array
        return torch.from_numpy(array)

    def _build_requests(
        self,
        job_id: str,
        subtask_ids: list[str],
        stages: tuple[str, ...],
        backend: str,
        library_path: str | None,
        allow_resume: bool,
    ) -> Iterable[ExecutionRequest]:
        bindings = {row["subtask_id"]: row for row in self.store.list_job_subtasks(job_id)}
        for subtask in self.store.list_subtasks(subtask_ids):
            if allow_resume and bindings.get(subtask.subtask_id) and bindings[subtask.subtask_id]["status"] == "COMPLETED":
                continue
            submodel = self.store.get_submodel(subtask.submodel_id)
            run_params = subtask.run_params
            yield ExecutionRequest(
                job_id=job_id,
                subtask_id=subtask.subtask_id,
                stages=stages,
                backend_name=backend,
                library_path=library_path,
                velocity_path=str(submodel.velocity_path),
                source=tuple(run_params["local_source"]),
                receivers=[tuple(item) for item in run_params["local_receivers"]],
                wavelet=list(run_params["wavelet"]),
                observed_data=run_params.get("observed_data"),
                dt=float(run_params["dt"]),
                dx=float(run_params["dx"]),
                dy=float(run_params["dy"]),
                dz=float(run_params["dz"]),
                save_forward_wavefield=subtask.checkpoint_policy.save_forward_wavefield,
                output_dir=str(self.root / "jobs" / job_id / subtask.subtask_id),
                reuse_forward=allow_resume,
            )

    def _filter_receivers(self, shot: Shot, rule: OffsetRule) -> tuple[list[Receiver], list[int]]:
        filtered: list[Receiver] = []
        indices: list[int] = []
        for index, receiver in enumerate(shot.receivers):
            offset = math.dist(shot.source, (receiver.x, receiver.y, receiver.z))
            if offset < rule.min_offset:
                continue
            if rule.max_offset is not None and offset > rule.max_offset:
                continue
            filtered.append(receiver)
            indices.append(index)
        return filtered, indices

    def _clip_horizons(
        self,
        horizons: list[Horizon],
        x0: int,
        x1: int,
        y0: int,
        y1: int,
        z0: int,
        z1: int,
    ) -> list[Horizon]:
        local_horizons: list[Horizon] = []
        for horizon in horizons:
            samples = [
                (x - x0, y - y0, z - z0)
                for x, y, z in horizon.samples
                if x0 <= x < x1 and y0 <= y < y1 and z0 <= z < z1
            ]
            if samples:
                local_horizons.append(Horizon(name=horizon.name, samples=samples))
        return local_horizons

    def _try_latest_dataset(self, kind: str):
        try:
            return self.store.latest_dataset(kind)
        except RuntimeError:
            return None

    @staticmethod
    def _decode_json(value: str) -> Any:
        return json.loads(value)
