from __future__ import annotations

import sys
from pathlib import Path

HEADER = r"""
typedef struct {
    int nx;
    int ny;
    int nz;
    float dx;
    float dy;
    float dz;
    float ox;
    float oy;
    float oz;
} wesi_grid_t;

typedef struct {
    wesi_grid_t grid;
    const float *velocity;
    int x0;
    int x1;
    int y0;
    int y1;
    int z0;
    int z1;
    int halo;
    int pml;
} wesi_submodel_t;

typedef struct {
    int source_x;
    int source_y;
    int source_z;
    int receiver_count;
    const int *receiver_xyz;
    int nt;
    const float *wavelet;
    const float *observed_data;
} wesi_shot_t;

typedef struct {
    int horizon_count;
    const float *samples;
    const int *counts;
} wesi_horizon_set_t;

typedef struct {
    int nt;
    float dt;
    int save_forward_wavefield;
    int checkpoint_stride;
    int threads;
} wesi_sim_params_t;

typedef struct {
    float *forward_wavefield;
    float *recorded_data;
    float *image;
} wesi_checkpoint_t;

int wesi_run_forward(
    const wesi_submodel_t *submodel,
    const wesi_shot_t *shot,
    const wesi_horizon_set_t *horizons,
    const wesi_sim_params_t *params,
    wesi_checkpoint_t *checkpoint
);

int wesi_run_rtm(
    const wesi_submodel_t *submodel,
    const wesi_shot_t *shot,
    const wesi_horizon_set_t *horizons,
    const wesi_sim_params_t *params,
    wesi_checkpoint_t *checkpoint
);
"""


def get_cdef() -> str:
    return HEADER



def main() -> None:
    try:
        from cffi import FFI  # type: ignore
    except ImportError as exc:
        raise SystemExit("cffi is required to build the Wesi C library") from exc

    root = Path(__file__).resolve().parents[3]
    ffi = FFI()
    ffi.cdef(HEADER)

    extra_compile_args = ["-O2"]
    extra_link_args: list[str] = []
    if sys.platform.startswith("linux"):
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")

    ffi.set_source(
        "wesi._wesi_cffi",
        '#include "wesi.h"',
        sources=[str(root / "csrc" / "wesi.c")],
        include_dirs=[str(root / "csrc")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    ffi.compile(verbose=True)


if __name__ == "__main__":
    main()
