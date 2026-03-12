# Wesi Development Guide

## 1. Document Purpose

This document is the long-term development guide for the Wesi project.
It is intended for:

- human developers joining or continuing the project
- Codex or other coding agents that need architectural context
- phased planning, task decomposition, and implementation review

The goal is to keep one stable description of:

- what has already been built
- what each current program/module is responsible for
- what still needs to be built
- what should be done in each development phase

This document should be updated whenever the architecture, module boundaries,
or development priorities change.

## 2. Project Goal

Wesi is a Linux-first 3D acoustic wave-equation simulation and RTM platform.

The intended technical direction is:

- Python as the main application language
- PySide6 for the desktop GUI
- VTK for 3D rendering and visualization
- C for simulation and RTM kernels
- SQLite + workspace files for project/job management
- Torch-compatible data interfaces for later AI/ML workflows

The system is organized around:

- project
- survey / velocity / horizons
- grid and work area generation
- submodel construction
- independent subtasks
- job orchestration
- result visualization

## 3. Current Repository Status

The repository is currently an MVP scaffold.

Already implemented:

- Python layered package structure
- C ABI header and reference C kernel
- NumPy reference backend for development/testing
- project workspace initialization
- velocity, shot, and horizon import
- grid construction from velocity volume
- submodel and subtask generation
- SQLite-backed job metadata
- local job execution and artifact persistence
- minimal PySide6 + VTK GUI shell
- basic tests

Not yet production-ready:

- real SEG-Y geometry/gather import pipeline
- robust C backend runtime path validation on Linux
- MPI executor
- mature GUI workflow and project editing
- advanced restart/checkpoint policies
- full visualization tools
- production numerical validation and performance tuning

## 4. Current Programs and Their Responsibilities

This section describes each existing program/module and why it exists.

### 4.1 Package Entry Layer

#### [src/wesi/__main__.py](/D:/Wesi/src/wesi/__main__.py)

Purpose:

- package entry point
- allows `python -m wesi` execution

Role in system:

- redirects to the CLI entry function

#### [src/wesi/cli.py](/D:/Wesi/src/wesi/cli.py)

Purpose:

- command-line launcher

Current responsibilities:

- initialize a project workspace
- launch the UI
- run a built-in demo workflow

Future responsibilities:

- add batch-mode job submission
- add headless project import/build commands
- add diagnostics/checkpoint inspection tools

### 4.2 Application Layer

#### [src/wesi/application/services.py](/D:/Wesi/src/wesi/application/services.py)

Purpose:

- main orchestration service for the whole application

Current responsibilities:

- create project metadata
- import velocity, shots, and horizons
- build grid from velocity
- build submodels and subtasks from survey + offset rules
- create jobs
- run jobs
- resume jobs
- run partial jobs
- load result as NumPy/Torch data

Why it matters:

- this is currently the highest-level application facade
- future GUI and CLI actions should call this service instead of bypassing it

Future responsibilities:

- stronger validation rules
- richer job filtering
- project open/save workflows
- better error classification
- MPI executor integration

#### [src/wesi/application/executors.py](/D:/Wesi/src/wesi/application/executors.py)

Purpose:

- execute subtasks using a backend runtime

Current responsibilities:

- define `ExecutionRequest`
- run subtasks locally
- manage subtask output directories
- persist recorded data, forward wavefield, image, and manifest
- support reuse of forward outputs during restart

Future responsibilities:

- proper thread/resource controls
- executor cancellation/pause
- timeout and retry policies
- MPI executor implementation
- better log and telemetry output

### 4.3 Domain Layer

#### [src/wesi/domain/models.py](/D:/Wesi/src/wesi/domain/models.py)

Purpose:

- define stable domain objects

Current responsibilities:

- grid specification
- offset rule
- receiver / shot
- horizon / horizon set
- checkpoint policy
- project config
- submodel
- subtask
- job config / job
- job artifact

Why it matters:

- this file defines the vocabulary used by the whole project
- later development should extend these models before adding ad hoc dictionaries

Future responsibilities:

- explicit survey model
- stage-specific run parameter models
- stronger serialization/versioning
- job state enums instead of string literals

### 4.4 Infrastructure Layer

#### [src/wesi/infrastructure/io.py](/D:/Wesi/src/wesi/infrastructure/io.py)

Purpose:

- project workspace I/O and normalized data persistence

Current responsibilities:

- create project directory layout
- read/write JSON
- save/load arrays
- import velocity sources
- import shot sources
- import horizon sources
- preserve raw imported files when applicable

Current limitations:

- SEG-Y support is only partial
- shot SEG-Y import is still a placeholder

Future responsibilities:

- full SEG-Y parsing for volumes and gathers
- sidecar schema validation
- data versioning and migration support
- artifact export helpers

#### [src/wesi/infrastructure/jobstore.py](/D:/Wesi/src/wesi/infrastructure/jobstore.py)

Purpose:

- SQLite metadata store

Current responsibilities:

- initialize schema
- persist project metadata
- persist datasets, grids, submodels, subtasks, jobs
- persist job-subtask bindings
- persist artifacts and checkpoints
- query current project/job state

Why it matters:

- this is the current source of truth for project/job metadata

Future responsibilities:

- schema versioning
- explicit migrations
- indexing and performance tuning
- query helpers for GUI summaries and analytics

### 4.5 Runtime / Bindings Layer

#### [src/wesi/bindings/base.py](/D:/Wesi/src/wesi/bindings/base.py)

Purpose:

- define backend abstraction

Current responsibilities:

- `SimulationBackend` interface
- `SimulationResult` result container

Future responsibilities:

- standard diagnostics contract
- stage-specific result types

#### [src/wesi/bindings/reference_backend.py](/D:/Wesi/src/wesi/bindings/reference_backend.py)

Purpose:

- NumPy reference implementation

Current responsibilities:

- development-friendly forward modeling
- development-friendly RTM imaging
- provide a runnable fallback when compiled C runtime is not available

Why it matters:

- keeps the project testable and explorable before Linux C toolchain integration is complete

Future responsibilities:

- remain a correctness reference backend
- provide comparison baselines against the C implementation

#### [src/wesi/bindings/api.py](/D:/Wesi/src/wesi/bindings/api.py)

Purpose:

- runtime backend selection and C ABI adapter

Current responsibilities:

- load compiled C backend if available
- fall back to NumPy backend in `auto` mode
- map NumPy buffers into the C ABI using `cffi`

Current limitations:

- runtime path needs Linux build/test validation
- horizons are not yet passed through in detail
- thread control is not yet wired from application config

Future responsibilities:

- production C backend validation
- backend capability reporting
- explicit CPU/GPU backend switching if introduced later

#### [src/wesi/bindings/build_ffi.py](/D:/Wesi/src/wesi/bindings/build_ffi.py)

Purpose:

- build script for the compiled C backend

Current responsibilities:

- expose the public C definitions to `cffi`
- compile `csrc/wesi.c`

Future responsibilities:

- Linux-only build profile handling
- release build integration
- debug/profile build options

### 4.6 UI Layer

#### [src/wesi/ui/main_window.py](/D:/Wesi/src/wesi/ui/main_window.py)

Purpose:

- MVP desktop shell

Current responsibilities:

- show datasets, subtasks, jobs
- build demo data
- run a demo job
- display logs
- pass volume/image data to the viewer

Current limitations:

- not yet a real project workflow UI
- only basic actions are exposed

Future responsibilities:

- full project lifecycle
- import forms and validation UI
- subtask filtering and job creation panels
- restart/hot-start controls
- failure inspection and rerun tools

#### [src/wesi/ui/viewer.py](/D:/Wesi/src/wesi/ui/viewer.py)

Purpose:

- VTK-backed volume viewer widget

Current responsibilities:

- show loaded 3D volume if VTK is installed
- degrade gracefully when VTK is unavailable

Future responsibilities:

- orthogonal slicing
- shot/receiver/horizon overlays
- submodel bounding box visualization
- image/velocity comparison views
- interactive picking and inspection

### 4.7 C Runtime Layer

#### [csrc/wesi.h](/D:/Wesi/csrc/wesi.h)

Purpose:

- public C ABI header

Current responsibilities:

- define stable structures:
  - `wesi_grid_t`
  - `wesi_submodel_t`
  - `wesi_shot_t`
  - `wesi_horizon_set_t`
  - `wesi_sim_params_t`
  - `wesi_checkpoint_t`
- declare:
  - `wesi_run_forward`
  - `wesi_run_rtm`

Why it matters:

- this header is the boundary contract between Python and C
- future C work should preserve ABI stability whenever possible

#### [csrc/wesi.c](/D:/Wesi/csrc/wesi.c)

Purpose:

- reference C kernel implementation

Current responsibilities:

- validate basic inputs
- perform simple 3D finite-difference forward propagation
- record receiver traces
- perform reverse-time imaging
- support OpenMP if available

Current limitations:

- MVP numerical implementation only
- not yet optimized for production RTM
- no advanced absorbing boundary model
- no advanced checkpoint compression or scheduling

Future responsibilities:

- production-grade numerical kernels
- more robust boundary conditions
- better memory strategy
- thread scaling and profiling

### 4.8 Test Layer

#### [tests/test_workflow.py](/D:/Wesi/tests/test_workflow.py)

Purpose:

- verify the end-to-end MVP workflow

Current coverage:

- project creation
- import pipeline
- grid build
- subtask build
- job execution
- artifact generation
- resume path

#### [tests/test_bindings.py](/D:/Wesi/tests/test_bindings.py)

Purpose:

- verify C ABI definitions exist in the exposed cdef

#### [tests/conftest.py](/D:/Wesi/tests/conftest.py)

Purpose:

- make `src/` importable during tests

## 5. Programs That Still Need To Be Implemented

This section lists important programs/modules that should exist in later phases.

### 5.1 Survey and Geometry Modules

Recommended future files:

- `src/wesi/domain/survey.py`
- `src/wesi/infrastructure/segy_geometry.py`
- `src/wesi/application/survey_service.py`

Purpose:

- formalize source/receiver geometry
- parse real SEG-Y gathers and geometry sidecars
- support line/patch organization and shot selection

### 5.2 Project Management Modules

Recommended future files:

- `src/wesi/application/project_service.py`
- `src/wesi/infrastructure/project_manifest.py`
- `src/wesi/infrastructure/migrations.py`

Purpose:

- open existing projects cleanly
- version project schema/layout
- perform automatic migrations

### 5.3 Job Control Modules

Recommended future files:

- `src/wesi/application/job_service.py`
- `src/wesi/application/checkpoint_service.py`
- `src/wesi/application/scheduler.py`

Purpose:

- separate orchestration concerns from the monolithic service
- add pause/resume/cancel/retry logic
- centralize hot-start and checkpoint policy

### 5.4 MPI and Cluster Execution

Recommended future files:

- `src/wesi/application/mpi_executor.py`
- `src/wesi/application/resource_model.py`
- `src/wesi/infrastructure/hostfile.py`

Purpose:

- support cluster or multi-node execution
- isolate MPI-specific launch logic
- model resources explicitly

### 5.5 Visualization Extensions

Recommended future files:

- `src/wesi/ui/project_panel.py`
- `src/wesi/ui/import_panel.py`
- `src/wesi/ui/job_panel.py`
- `src/wesi/ui/log_panel.py`
- `src/wesi/ui/scene_overlays.py`

Purpose:

- split the MVP window into maintainable UI modules
- support production workflows

### 5.6 Torch and Data Science Integration

Recommended future files:

- `src/wesi/application/tensor_service.py`
- `src/wesi/bindings/dlpack_bridge.py`

Purpose:

- centralize tensor conversions
- later support zero-copy or GPU-friendly data exchange

### 5.7 Numerical Runtime Growth

Recommended future files:

- `csrc/fd3d_acoustic.c`
- `csrc/rtm3d_acoustic.c`
- `csrc/checkpoint.c`
- `csrc/runtime_types.h`

Purpose:

- split the current single C source into focused runtime modules
- make future optimization and testing easier

## 6. Recommended Near-Term Architecture Refactor

The current scaffold is good for MVP exploration, but later work should move
toward the following structure:

```text
src/wesi/
  application/
    project_service.py
    data_service.py
    model_builder.py
    job_service.py
    checkpoint_service.py
    executors/
      local.py
      mpi.py
  domain/
    models.py
    survey.py
    states.py
  infrastructure/
    io.py
    segy_geometry.py
    jobstore.py
    migrations.py
  bindings/
    base.py
    api.py
    reference_backend.py
    dlpack_bridge.py
  ui/
    main_window.py
    project_panel.py
    import_panel.py
    viewer.py
    job_panel.py
```

This is not required immediately, but it is the recommended medium-term shape.

## 7. Development Phases

This section is the main execution plan for ongoing development.

### Phase 0 - Environment and Build Baseline

Goal:

- make the repository reproducible on Linux

Tasks:

- define official Linux Python version
- define build dependencies for C and `cffi`
- verify editable install workflow
- verify C backend compilation workflow
- document optional dependency sets:
  - UI
  - VTK
  - SEG-Y
  - Torch

Deliverables:

- updated environment instructions
- verified Linux build process
- baseline CI plan

### Phase 1 - Data and Project Foundation

Goal:

- make project data ingestion and persistence stable

Tasks:

- formalize project workspace schema
- formalize dataset metadata fields
- improve input validation
- add project open/load support
- support sidecar schema checks
- add project migration/version field

Deliverables:

- stable project workspace layout
- stable metadata schema
- reliable import/open workflow

### Phase 2 - Survey and Subtask Construction

Goal:

- make submodel generation reliable for real data

Tasks:

- implement real geometry parsing
- support line/patch organization
- validate source/receiver mapping
- validate offset-based filtering
- add deterministic subtask generation reports
- persist submodel lineage metadata

Deliverables:

- real survey ingestion pipeline
- traceable subtask generation
- better geometry diagnostics

### Phase 3 - Numerical Runtime Maturity

Goal:

- move from MVP kernel to dependable numerical runtime

Tasks:

- split C runtime into focused modules
- validate C vs NumPy outputs on small models
- improve boundaries/PML behavior
- expose runtime thread controls
- measure memory use and performance
- add kernel-level tests

Deliverables:

- validated C runtime
- documented numerical assumptions
- small regression suite for forward + RTM

### Phase 4 - Job Management and Restart

Goal:

- turn execution into a real job system

Tasks:

- add explicit job state transitions
- add pause/cancel/retry semantics
- add partial stage execution UI/CLI
- improve checkpoint metadata
- implement hot-start logic more explicitly
- add failure recovery scenarios

Deliverables:

- stable restart/recovery model
- inspectable job histories
- reliable partial reruns

### Phase 5 - GUI Workflow Completion

Goal:

- make the desktop application usable for real operator workflows

Tasks:

- split the current monolithic window into panels
- add project browser
- add import wizard/forms
- add subtask selection UI
- add job configuration and launch UI
- add failure/log inspection tools
- add result browsing UI

Deliverables:

- full project workflow GUI
- clearer user path from import to imaging result

### Phase 6 - Visualization Upgrade

Goal:

- make visualization useful for analysis, not just demo display

Tasks:

- add slice views
- add 3D overlays for sources/receivers
- add horizon overlays
- add submodel bounding boxes
- add color map controls
- add comparison views for velocity/image/checkpoint products

Deliverables:

- production-oriented interpretation display

### Phase 7 - Torch Integration

Goal:

- make data exchange with ML pipelines clean and stable

Tasks:

- centralize NumPy/Torch conversions
- define tensor loading APIs
- define artifact-to-tensor conventions
- optionally add DLPack bridge
- define CPU/GPU behavior expectations

Deliverables:

- stable tensor interface for downstream workflows

### Phase 8 - MPI and Advanced Parallelism

Goal:

- support large-scale compute execution

Tasks:

- design MPI executor contract
- map subtasks to ranks or job groups
- define distributed logging and checkpoint policy
- define local vs MPI execution configuration
- add cluster launch tests

Deliverables:

- first working MPI execution mode

### Phase 9 - Production Hardening

Goal:

- make the platform maintainable and dependable

Tasks:

- add CI matrix
- add smoke tests for optional feature sets
- add profiling scripts
- add release packaging strategy
- add project migration tests
- add performance benchmark cases

Deliverables:

- release-ready engineering baseline

## 8. Suggested Development Order Right Now

If development continues from the current repository state, the recommended
next order is:

1. stabilize Linux build and runtime verification
2. improve project open/load and schema versioning
3. implement real SEG-Y survey import
4. strengthen subtask generation validation
5. validate and improve C backend behavior
6. improve restart/checkpoint logic
7. split GUI into maintainable panels
8. expand VTK analytical views
9. add Torch service and optional DLPack path
10. implement MPI mode

## 9. Rules for Future Development

To keep the repository coherent, future development should follow these rules:

- keep Python as the orchestration layer
- keep C as the numerical kernel layer
- do not let C access the project database or workspace policy directly
- prefer typed domain models over loose dictionaries
- keep artifacts and checkpoints discoverable from SQLite metadata
- keep each subtask independently runnable
- preserve restartability as a first-class requirement
- make all new GUI actions route through application services
- prefer additive, versioned schema evolution over ad hoc breaking changes

## 10. Guidance for Codex or Other Agents

When using this repository for automated analysis or coding, the preferred
mental model is:

- `services.py` is the current high-level entry point
- `jobstore.py` is the metadata source of truth
- `io.py` is the normalized data boundary
- `api.py` and `wesi.h` define the Python/C runtime boundary
- `reference_backend.py` is the correctness-oriented fallback runtime
- `wesi.c` is the current reference C implementation, not yet the final kernel

When planning new work, agents should:

- first decide whether the change belongs to domain, application, infrastructure, UI, or C runtime
- avoid bypassing the service layer unless explicitly refactoring it
- preserve the independent-subtask execution model
- update this document when adding major modules or changing development order

## 11. Maintenance of This Document

This document should be updated when any of the following change:

- package/module layout
- job execution model
- runtime ABI
- project workspace layout
- development priorities
- phase order

Recommended update rule:

- if a PR adds a new top-level module or changes a key workflow, update this file in the same change
