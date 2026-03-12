from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from .base import SimulationBackend, SimulationResult
from .reference_backend import NumPyReferenceBackend


class CffiKernelBackend(SimulationBackend):
    name = "cffi"

    def __init__(self, library_path: str | Path | None = None) -> None:
        self._ffi = None
        self._library = None
        self._library_path = library_path
        self._load_library()

    def _load_library(self) -> None:
        if self._library_path is None:
            try:
                module = importlib.import_module("wesi._wesi_cffi")
            except ImportError as exc:
                raise RuntimeError("Compiled wesi._wesi_cffi module not found. Run `python -m wesi.bindings.build_ffi` on Linux.") from exc
            self._ffi = module.ffi
            self._library = module.lib
            return
        try:
            from cffi import FFI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("cffi is not installed") from exc
        from .build_ffi import get_cdef

        ffi = FFI()
        ffi.cdef(get_cdef())
        self._ffi = ffi
        self._library = ffi.dlopen(str(self._library_path))

    def run_forward(
        self,
        velocity: np.ndarray,
        source: tuple[int, int, int],
        receivers: list[tuple[int, int, int]],
        wavelet: np.ndarray,
        save_wavefield: bool,
        dt: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> SimulationResult:
        ffi = self._ffi
        lib = self._library
        assert ffi is not None and lib is not None
        velocity = np.ascontiguousarray(velocity.astype(np.float32))
        wavelet = np.ascontiguousarray(wavelet.astype(np.float32))
        receiver_xyz = np.ascontiguousarray(np.asarray(receivers, dtype=np.int32).reshape(-1))
        recorded = np.zeros((wavelet.shape[0], len(receivers)), dtype=np.float32)
        forward = np.zeros((wavelet.shape[0], *velocity.shape), dtype=np.float32) if save_wavefield else None
        status = lib.wesi_run_forward(
            self._build_submodel(ffi, velocity, dx, dy, dz),
            self._build_shot(ffi, source, receiver_xyz, wavelet, None),
            self._build_horizons(ffi),
            self._build_params(ffi, int(wavelet.shape[0]), dt, save_wavefield),
            self._build_checkpoint(ffi, recorded, forward, None),
        )
        if status != 0:
            raise RuntimeError(f"wesi_run_forward failed with code {status}")
        return SimulationResult(
            recorded_data=recorded,
            forward_wavefield=forward,
            diagnostics={"backend": self.name, "status": status},
        )

    def run_rtm(
        self,
        velocity: np.ndarray,
        source: tuple[int, int, int],
        receivers: list[tuple[int, int, int]],
        wavelet: np.ndarray,
        observed_data: np.ndarray,
        forward_wavefield: np.ndarray | None,
        dt: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> SimulationResult:
        ffi = self._ffi
        lib = self._library
        assert ffi is not None and lib is not None
        velocity = np.ascontiguousarray(velocity.astype(np.float32))
        wavelet = np.ascontiguousarray(wavelet.astype(np.float32))
        observed = np.ascontiguousarray(observed_data.astype(np.float32))
        receiver_xyz = np.ascontiguousarray(np.asarray(receivers, dtype=np.int32).reshape(-1))
        forward = np.ascontiguousarray(forward_wavefield.astype(np.float32)) if forward_wavefield is not None else None
        image = np.zeros_like(velocity, dtype=np.float32)
        recorded = observed.copy()
        status = lib.wesi_run_rtm(
            self._build_submodel(ffi, velocity, dx, dy, dz),
            self._build_shot(ffi, source, receiver_xyz, wavelet, observed),
            self._build_horizons(ffi),
            self._build_params(ffi, int(wavelet.shape[0]), dt, forward is not None),
            self._build_checkpoint(ffi, recorded, forward, image),
        )
        if status != 0:
            raise RuntimeError(f"wesi_run_rtm failed with code {status}")
        return SimulationResult(image=image, forward_wavefield=forward, diagnostics={"backend": self.name, "status": status})

    def _build_submodel(self, ffi, velocity: np.ndarray, dx: float, dy: float, dz: float):
        nz, ny, nx = velocity.shape
        return ffi.new(
            "wesi_submodel_t *",
            {
                "grid": {
                    "nx": nx,
                    "ny": ny,
                    "nz": nz,
                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                    "ox": 0.0,
                    "oy": 0.0,
                    "oz": 0.0,
                },
                "velocity": ffi.from_buffer("float *", velocity),
                "x0": 0,
                "x1": nx,
                "y0": 0,
                "y1": ny,
                "z0": 0,
                "z1": nz,
                "halo": 0,
                "pml": 0,
            },
        )

    def _build_shot(self, ffi, source, receiver_xyz, wavelet, observed):
        observed_ptr = ffi.NULL if observed is None else ffi.from_buffer("float *", observed)
        return ffi.new(
            "wesi_shot_t *",
            {
                "source_x": int(source[0]),
                "source_y": int(source[1]),
                "source_z": int(source[2]),
                "receiver_count": int(receiver_xyz.size // 3),
                "receiver_xyz": ffi.from_buffer("int *", receiver_xyz),
                "nt": int(wavelet.shape[0]),
                "wavelet": ffi.from_buffer("float *", wavelet),
                "observed_data": observed_ptr,
            },
        )

    def _build_horizons(self, ffi):
        return ffi.new(
            "wesi_horizon_set_t *",
            {"horizon_count": 0, "samples": ffi.NULL, "counts": ffi.NULL},
        )

    def _build_params(self, ffi, nt: int, dt: float, save_wavefield: bool):
        return ffi.new(
            "wesi_sim_params_t *",
            {
                "nt": nt,
                "dt": dt,
                "save_forward_wavefield": int(save_wavefield),
                "checkpoint_stride": 1,
                "threads": 1,
            },
        )

    def _build_checkpoint(self, ffi, recorded, forward, image):
        return ffi.new(
            "wesi_checkpoint_t *",
            {
                "forward_wavefield": ffi.NULL if forward is None else ffi.from_buffer("float *", forward),
                "recorded_data": ffi.from_buffer("float *", recorded),
                "image": ffi.NULL if image is None else ffi.from_buffer("float *", image),
            },
        )



def backend_from_name(name: str = "auto", library_path: str | Path | None = None) -> SimulationBackend:
    lowered = name.lower()
    if lowered == "auto":
        try:
            return CffiKernelBackend(library_path=None)
        except RuntimeError:
            return NumPyReferenceBackend()
    if lowered in {"numpy", "reference"}:
        return NumPyReferenceBackend()
    if lowered in {"c", "cffi"}:
        return CffiKernelBackend(library_path)
    raise ValueError(f"Unknown backend: {name}")
