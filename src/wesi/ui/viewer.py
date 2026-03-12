from __future__ import annotations

import numpy as np

try:
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
except ImportError as exc:  # pragma: no cover - optional UI dependency
    raise RuntimeError("PySide6 is required for the UI") from exc

try:  # pragma: no cover - optional VTK dependency
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:  # pragma: no cover - optional VTK dependency
    vtk = None
    QVTKRenderWindowInteractor = None


class VolumeViewerWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        if vtk is None or QVTKRenderWindowInteractor is None:
            self._placeholder = QLabel("VTK is not installed. Volume rendering is unavailable.")
            self._layout.addWidget(self._placeholder)
            self._vtk_widget = None
            self._renderer = None
            return
        self._vtk_widget = QVTKRenderWindowInteractor(self)
        self._layout.addWidget(self._vtk_widget)
        self._renderer = vtk.vtkRenderer()
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)
        self._renderer.SetBackground(0.08, 0.1, 0.14)

    def show_volume(self, volume: np.ndarray) -> None:
        if self._vtk_widget is None or self._renderer is None or vtk is None:
            if hasattr(self, "_placeholder"):
                self._placeholder.setText(f"Volume loaded with shape {tuple(volume.shape)}, but VTK is unavailable.")
            return
        data = np.ascontiguousarray(volume.astype(np.float32))
        importer = vtk.vtkImageImport()
        importer.CopyImportVoidPointer(data.tobytes(), data.nbytes)
        importer.SetDataScalarTypeToFloat()
        importer.SetNumberOfScalarComponents(1)
        nz, ny, nx = data.shape
        importer.SetDataExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
        importer.SetWholeExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
        importer.Update()

        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputConnection(importer.GetOutputPort())
        color = vtk.vtkColorTransferFunction()
        color.AddRGBPoint(float(data.min()), 0.1, 0.1, 0.3)
        color.AddRGBPoint(float(data.mean()), 0.8, 0.8, 0.3)
        color.AddRGBPoint(float(data.max()), 0.95, 0.95, 0.95)
        opacity = vtk.vtkPiecewiseFunction()
        opacity.AddPoint(float(data.min()), 0.0)
        opacity.AddPoint(float(data.mean()), 0.08)
        opacity.AddPoint(float(data.max()), 0.25)
        prop = vtk.vtkVolumeProperty()
        prop.SetColor(color)
        prop.SetScalarOpacity(opacity)
        prop.ShadeOff()
        volume_actor = vtk.vtkVolume()
        volume_actor.SetMapper(mapper)
        volume_actor.SetProperty(prop)

        self._renderer.RemoveAllViewProps()
        self._renderer.AddVolume(volume_actor)
        self._renderer.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()
