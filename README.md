# Wesi

Wesi is a Linux-first MVP for 3D acoustic wave-equation simulation and RTM.
The application uses Python for orchestration, PySide6 for the desktop UI,
VTK for 3D visualization, and a pure C ABI for simulation kernels.

## Highlights

- Python application layers for project, data, and job orchestration
- Stable C ABI designed for minimal parameter passing
- SQLite-backed job metadata with restart and partial-run support
- Independent subtasks built from shots, offsets, and submodels
- Optional Torch tensor bridge for post-processing workflows
- Reference NumPy backend for development when the C toolchain is unavailable

## Layout

```text
src/wesi/
  application/     service layer and executors
  bindings/        C ABI loader, cffi builder, NumPy reference backend
  domain/          typed domain models
  infrastructure/  SQLite store and data import/export helpers
  ui/              PySide6 + VTK desktop shell
csrc/
  wesi.h           public C ABI header
  wesi.c           reference C kernel implementation
tests/
  end-to-end smoke tests for the MVP workflow
```

## Quick Start

1. Create a Python environment on Linux with `numpy`, `cffi`, and `pytest`.
2. Install the package in editable mode: `pip install -e .`
3. Run the test suite: `pytest`
4. Launch the desktop shell: `python -m wesi ui`

The UI and SEG-Y import paths are optional dependency features. The repository
includes a JSON/CSV/NPY development path so the full workflow can be exercised
without seismic vendor data or compiled extensions.
