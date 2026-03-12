from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - optional UI dependency
    raise RuntimeError("PySide6 is required for the UI") from exc

from wesi.application.services import WesiService

from .viewer import VolumeViewerWidget


class WesiMainWindow(QMainWindow):
    def __init__(self, service: WesiService) -> None:
        super().__init__()
        self.service = service
        self.setWindowTitle("Wesi MVP")
        self.resize(1400, 900)

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        toolbar = QHBoxLayout()
        layout.addLayout(toolbar)
        self.refresh_button = QPushButton("Refresh")
        self.demo_button = QPushButton("Build Demo")
        self.run_button = QPushButton("Run Demo Job")
        toolbar.addWidget(self.refresh_button)
        toolbar.addWidget(self.demo_button)
        toolbar.addWidget(self.run_button)
        toolbar.addStretch(1)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, stretch=1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Datasets"))
        self.dataset_list = QListWidget()
        left_layout.addWidget(self.dataset_list)
        left_layout.addWidget(QLabel("Subtasks"))
        self.subtask_list = QListWidget()
        left_layout.addWidget(self.subtask_list)
        left_layout.addWidget(QLabel("Jobs"))
        self.job_list = QListWidget()
        left_layout.addWidget(self.job_list)
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.viewer = VolumeViewerWidget()
        right_layout.addWidget(self.viewer, stretch=1)
        right_layout.addWidget(QLabel("Logs"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        right_layout.addWidget(self.log_view, stretch=1)
        splitter.addWidget(right_panel)
        splitter.setSizes([380, 1020])

        self.refresh_button.clicked.connect(self.refresh)
        self.demo_button.clicked.connect(self.build_demo)
        self.run_button.clicked.connect(self.run_demo_job)

        self.refresh()

    def refresh(self) -> None:
        self.dataset_list.clear()
        self.subtask_list.clear()
        self.job_list.clear()
        for row in self.service.store.list_datasets():
            self.dataset_list.addItem(f"{row['kind']}: {row['dataset_id']}")
        for subtask in self.service.store.list_subtasks():
            self.subtask_list.addItem(f"{subtask.subtask_id} [{subtask.status}]")
        with self.service.store.connect() as connection:
            for row in connection.execute("SELECT job_id, status FROM jobs ORDER BY rowid DESC"):
                self.job_list.addItem(f"{row['job_id']} [{row['status']}]")

    def build_demo(self) -> None:
        if not self._project_exists():
            self.service.create_project("Demo Project")
        velocity = np.full((24, 24, 24), 1800.0, dtype=np.float32)
        velocity[12:, 10:14, 10:14] = 2200.0
        shots = [
            {
                "shot_id": "shot-001",
                "source": [12, 12, 2],
                "receivers": [{"x": x, "y": 12, "z": 2} for x in range(4, 20)],
                "wavelet": [0.0, 0.5, 1.0, 0.5, 0.0, -0.25, -0.1, 0.0],
            }
        ]
        horizons = {
            "horizons": [
                {"name": "target", "samples": [[x, y, 12.0] for x in range(6, 18) for y in range(6, 18)]}
            ]
        }
        velocity_id = self.service.import_velocity(velocity, name="demo-velocity")
        shots_id = self.service.import_shots(shots, name="demo-shots")
        horizon_path = Path(tempfile.gettempdir()) / "wesi_demo_horizons.json"
        horizon_path.write_text(__import__("json").dumps(horizons), encoding="utf-8")
        self.service.import_horizons(horizon_path, name="demo-horizons")
        self.service.build_grid(velocity_id=velocity_id)
        self.service.build_subtasks(velocity_id=velocity_id, shot_dataset_id=shots_id)
        self.viewer.show_volume(velocity)
        self._log("Demo project created")
        self.refresh()

    def run_demo_job(self) -> None:
        subtasks = [subtask.subtask_id for subtask in self.service.store.list_subtasks()]
        if not subtasks:
            QMessageBox.warning(self, "Wesi", "No subtasks available. Build the demo project first.")
            return
        job_id = self.service.create_job(subtask_ids=subtasks, max_workers=1)
        result = self.service.run_job(job_id, backend="numpy")
        self._log(f"Job {job_id} finished with status {result['status']}")
        artifacts = self.service.store.list_artifacts(job_id)
        image_rows = [row for row in artifacts if row["artifact_type"] == "image"]
        if image_rows:
            image = np.load(image_rows[0]["path"])
            self.viewer.show_volume(image)
        self.refresh()

    def _project_exists(self) -> bool:
        try:
            self.service.store.get_project()
        except RuntimeError:
            return False
        return True

    def _log(self, message: str) -> None:
        self.log_view.appendPlainText(message)



def launch(service: WesiService) -> int:
    app = QApplication.instance() or QApplication([])
    window = WesiMainWindow(service)
    window.show()
    return app.exec()
