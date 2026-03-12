from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from wesi.domain.models import CheckpointPolicy, GridSpec, Job, JobArtifact, JobConfig, ProjectConfig, Submodel, Subtask


JSONDict = dict[str, Any]


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.db_path = self.root / "project.db"

    def initialize(self) -> None:
        with self.connect() as connection:
            cursor = connection.cursor()
            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS project (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    root TEXT NOT NULL,
                    spacing_json TEXT NOT NULL,
                    origin_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    raw_path TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS grids (
                    grid_id TEXT PRIMARY KEY,
                    velocity_id TEXT NOT NULL,
                    spec_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS submodels (
                    submodel_id TEXT PRIMARY KEY,
                    velocity_id TEXT NOT NULL,
                    shot_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS subtasks (
                    subtask_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    shot_id TEXT NOT NULL,
                    submodel_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_subtasks (
                    job_id TEXT NOT NULL,
                    subtask_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    PRIMARY KEY (job_id, subtask_id)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    job_id TEXT NOT NULL,
                    subtask_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (job_id, subtask_id, artifact_type, path)
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    job_id TEXT NOT NULL,
                    subtask_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (job_id, subtask_id, stage)
                );
                """
            )
            connection.commit()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def save_project(self, project: ProjectConfig) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO project(project_id, name, root, spacing_json, origin_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    project.project_id,
                    project.name,
                    str(project.root),
                    json.dumps(project.spacing),
                    json.dumps(project.origin),
                    json.dumps(project.metadata),
                ),
            )
            connection.commit()

    def get_project(self) -> sqlite3.Row:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM project LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("Project metadata not initialized")
        return row

    def save_dataset(self, dataset_id: str, kind: str, path: Path, raw_path: Path | None, metadata: JSONDict) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO datasets(dataset_id, kind, path, raw_path, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (dataset_id, kind, str(path), str(raw_path) if raw_path else None, json.dumps(metadata)),
            )
            connection.commit()

    def get_dataset(self, dataset_id: str) -> sqlite3.Row:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise KeyError(dataset_id)
        return row

    def latest_dataset(self, kind: str) -> sqlite3.Row:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE kind = ? ORDER BY rowid DESC LIMIT 1", (kind,)
            ).fetchone()
        if row is None:
            raise RuntimeError(f"No dataset registered for kind={kind}")
        return row

    def list_datasets(self, kind: str | None = None) -> list[sqlite3.Row]:
        with self.connect() as connection:
            if kind is None:
                rows = connection.execute("SELECT * FROM datasets ORDER BY rowid").fetchall()
            else:
                rows = connection.execute("SELECT * FROM datasets WHERE kind = ? ORDER BY rowid", (kind,)).fetchall()
        return rows

    def save_grid(self, grid_id: str, velocity_id: str, grid: GridSpec) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO grids(grid_id, velocity_id, spec_json) VALUES (?, ?, ?)",
                (grid_id, velocity_id, json.dumps(grid.as_dict())),
            )
            connection.commit()

    def latest_grid(self) -> tuple[str, GridSpec]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM grids ORDER BY rowid DESC LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("Grid has not been built")
        spec = json.loads(row["spec_json"])
        return row["grid_id"], GridSpec(**spec)

    def save_submodel(self, submodel: Submodel) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO submodels(submodel_id, velocity_id, shot_id, payload_json) VALUES (?, ?, ?, ?)",
                (submodel.submodel_id, submodel.velocity_id, submodel.shot_id, json.dumps(submodel.as_dict())),
            )
            connection.commit()

    def get_submodel(self, submodel_id: str) -> Submodel:
        with self.connect() as connection:
            row = connection.execute("SELECT payload_json FROM submodels WHERE submodel_id = ?", (submodel_id,)).fetchone()
        if row is None:
            raise KeyError(submodel_id)
        payload = json.loads(row["payload_json"])
        payload["velocity_path"] = Path(payload["velocity_path"])
        payload["horizon_path"] = Path(payload["horizon_path"]) if payload["horizon_path"] else None
        payload["grid"] = GridSpec(**payload["grid"])
        return Submodel(**payload)

    def save_subtask(self, subtask: Subtask) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO subtasks(subtask_id, project_id, shot_id, submodel_id, stage, status, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subtask.subtask_id,
                    subtask.project_id,
                    subtask.shot_id,
                    subtask.submodel_id,
                    subtask.stage,
                    subtask.status,
                    json.dumps(subtask.as_dict()),
                ),
            )
            connection.commit()

    def get_subtask(self, subtask_id: str) -> Subtask:
        with self.connect() as connection:
            row = connection.execute("SELECT payload_json FROM subtasks WHERE subtask_id = ?", (subtask_id,)).fetchone()
        if row is None:
            raise KeyError(subtask_id)
        return self._subtask_from_payload(row["payload_json"])

    def list_subtasks(self, subtask_ids: list[str] | None = None) -> list[Subtask]:
        with self.connect() as connection:
            if subtask_ids:
                placeholders = ",".join("?" for _ in subtask_ids)
                rows = connection.execute(
                    f"SELECT payload_json FROM subtasks WHERE subtask_id IN ({placeholders}) ORDER BY rowid",
                    tuple(subtask_ids),
                ).fetchall()
            else:
                rows = connection.execute("SELECT payload_json FROM subtasks ORDER BY rowid").fetchall()
        return [self._subtask_from_payload(row["payload_json"]) for row in rows]

    def update_subtask_status(self, subtask_id: str, status: str) -> None:
        with self.connect() as connection:
            row = connection.execute("SELECT payload_json FROM subtasks WHERE subtask_id = ?", (subtask_id,)).fetchone()
            if row is None:
                raise KeyError(subtask_id)
            payload = json.loads(row["payload_json"])
            payload["status"] = status
            connection.execute(
                "UPDATE subtasks SET status = ?, payload_json = ? WHERE subtask_id = ?",
                (status, json.dumps(payload), subtask_id),
            )
            connection.commit()

    def create_job(self, job: Job) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO jobs(job_id, project_id, status, payload_json) VALUES (?, ?, ?, ?)",
                (job.job_id, job.project_id, job.status, json.dumps(job.as_dict())),
            )
            connection.commit()

    def get_job(self, job_id: str) -> Job:
        with self.connect() as connection:
            row = connection.execute("SELECT payload_json, status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        payload = json.loads(row["payload_json"])
        payload["config"] = JobConfig(**payload["config"])
        payload["status"] = row["status"]
        return Job(**payload)

    def update_job_status(self, job_id: str, status: str) -> None:
        with self.connect() as connection:
            row = connection.execute("SELECT payload_json FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            payload = json.loads(row["payload_json"])
            payload["status"] = status
            connection.execute("UPDATE jobs SET status = ?, payload_json = ? WHERE job_id = ?", (status, json.dumps(payload), job_id))
            connection.commit()

    def bind_job_subtask(self, job_id: str, subtask_id: str, status: str = "READY", error_message: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO job_subtasks(job_id, subtask_id, status, error_message)
                VALUES (?, ?, ?, ?)
                """,
                (job_id, subtask_id, status, error_message),
            )
            connection.commit()

    def update_job_subtask(self, job_id: str, subtask_id: str, status: str, error_message: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE job_subtasks SET status = ?, error_message = ? WHERE job_id = ? AND subtask_id = ?",
                (status, error_message, job_id, subtask_id),
            )
            connection.commit()

    def list_job_subtasks(self, job_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM job_subtasks WHERE job_id = ? ORDER BY rowid", (job_id,)
            ).fetchall()

    def register_artifact(self, artifact: JobArtifact) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO artifacts(job_id, subtask_id, artifact_type, path, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (
                    artifact.job_id,
                    artifact.subtask_id,
                    artifact.artifact_type,
                    str(artifact.path),
                    json.dumps(artifact.metadata),
                ),
            )
            connection.commit()

    def list_artifacts(self, job_id: str, subtask_id: str | None = None) -> list[sqlite3.Row]:
        with self.connect() as connection:
            if subtask_id is None:
                return connection.execute("SELECT * FROM artifacts WHERE job_id = ? ORDER BY rowid", (job_id,)).fetchall()
            return connection.execute(
                "SELECT * FROM artifacts WHERE job_id = ? AND subtask_id = ? ORDER BY rowid",
                (job_id, subtask_id),
            ).fetchall()

    def register_checkpoint(self, job_id: str, subtask_id: str, stage: str, path: Path, metadata: JSONDict) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO checkpoints(job_id, subtask_id, stage, path, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (job_id, subtask_id, stage, str(path), json.dumps(metadata)),
            )
            connection.commit()

    def get_checkpoint(self, job_id: str, subtask_id: str, stage: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM checkpoints WHERE job_id = ? AND subtask_id = ? AND stage = ?",
                (job_id, subtask_id, stage),
            ).fetchone()

    @staticmethod
    def _subtask_from_payload(payload_json: str) -> Subtask:
        payload = json.loads(payload_json)
        payload["checkpoint_policy"] = CheckpointPolicy(**payload["checkpoint_policy"])
        return Subtask(**payload)
