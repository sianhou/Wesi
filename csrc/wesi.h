#ifndef WESI_H
#define WESI_H

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif
