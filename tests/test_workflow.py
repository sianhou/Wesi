from __future__ import annotations

from pathlib import Path

import numpy as np

from wesi.application.services import WesiService
from wesi.domain.models import Horizon, HorizonSet, OffsetRule


def test_end_to_end_numpy_workflow(tmp_path: Path) -> None:
    root = tmp_path / "project"
    service = WesiService(root)
    service.create_project("Test Project", spacing=(10.0, 10.0, 10.0))

    velocity = np.full((16, 16, 16), 1800.0, dtype=np.float32)
    velocity[8:, 7:9, 7:9] = 2200.0
    shots = [
        {
            "shot_id": "shot-001",
            "source": [8, 8, 2],
            "receivers": [{"x": x, "y": 8, "z": 2} for x in range(4, 12)],
            "wavelet": [0.0, 0.5, 1.0, 0.5, 0.0, -0.25, -0.1, 0.0],
        }
    ]
    horizons = HorizonSet(horizons=[Horizon(name="target", samples=[(8, 8, 8.0), (9, 8, 8.0)])])

    velocity_id = service.import_velocity(velocity)
    shots_id = service.import_shots(shots)
    service.import_horizons(horizons)
    grid = service.build_grid(velocity_id=velocity_id)
    assert (grid.nx, grid.ny, grid.nz) == (16, 16, 16)

    subtask_ids = service.build_subtasks(
        velocity_id=velocity_id,
        shot_dataset_id=shots_id,
        offset_rule=OffsetRule(inline_padding=2, crossline_padding=2, depth_padding=2, pml=2),
    )
    assert len(subtask_ids) == 1

    job_id = service.create_job(subtask_ids=subtask_ids, max_workers=1)
    result = service.run_job(job_id, backend="numpy")
    assert result["status"] == "COMPLETED"

    artifacts = service.store.list_artifacts(job_id, subtask_ids[0])
    artifact_types = {row["artifact_type"] for row in artifacts}
    assert {"recorded", "forward_wavefield", "image"}.issubset(artifact_types)

    image = service.load_result_as_tensor(job_id, subtask_ids[0], "image")
    shape = tuple(image.shape) if hasattr(image, "shape") else tuple(image.size())
    assert len(shape) == 3

    resumed = service.resume_job(job_id, backend="numpy")
    assert resumed["status"] == "COMPLETED"


def test_import_horizons_from_csv(tmp_path: Path) -> None:
    root = tmp_path / "project"
    service = WesiService(root)
    service.create_project("CSV Horizon Project")
    csv_path = tmp_path / "horizons.csv"
    csv_path.write_text("name,x,y,z\nA,1,1,5\nA,2,1,5\nB,3,3,7\n", encoding="utf-8")

    horizon_id = service.import_horizons(csv_path)
    row = service.store.get_dataset(horizon_id)
    assert Path(row["path"]).exists()
