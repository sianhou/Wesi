#include "wesi.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static size_t wesi_index3(int x, int y, int z, int nx, int ny) {
    return (size_t)z * (size_t)ny * (size_t)nx + (size_t)y * (size_t)nx + (size_t)x;
}

static int wesi_validate(const wesi_submodel_t *submodel, const wesi_shot_t *shot, const wesi_sim_params_t *params, const wesi_checkpoint_t *checkpoint) {
    if (submodel == NULL || shot == NULL || params == NULL || checkpoint == NULL) {
        return -1;
    }
    if (submodel->velocity == NULL || shot->wavelet == NULL) {
        return -2;
    }
    if (submodel->grid.nx <= 2 || submodel->grid.ny <= 2 || submodel->grid.nz <= 2) {
        return -3;
    }
    if (params->nt <= 0 || shot->nt != params->nt) {
        return -4;
    }
    if (checkpoint->recorded_data == NULL) {
        return -5;
    }
    return 0;
}

static void wesi_laplacian(const float *curr, float *lap, int nx, int ny, int nz, float inv_dx2, float inv_dy2, float inv_dz2) {
    int x;
    int y;
    int z;
    #pragma omp parallel for private(z, y, x) collapse(2)
    for (z = 1; z < nz - 1; ++z) {
        for (y = 1; y < ny - 1; ++y) {
            for (x = 1; x < nx - 1; ++x) {
                size_t idx = wesi_index3(x, y, z, nx, ny);
                float center = curr[idx];
                lap[idx] =
                    (curr[wesi_index3(x + 1, y, z, nx, ny)] - 2.0f * center + curr[wesi_index3(x - 1, y, z, nx, ny)]) * inv_dx2 +
                    (curr[wesi_index3(x, y + 1, z, nx, ny)] - 2.0f * center + curr[wesi_index3(x, y - 1, z, nx, ny)]) * inv_dy2 +
                    (curr[wesi_index3(x, y, z + 1, nx, ny)] - 2.0f * center + curr[wesi_index3(x, y, z - 1, nx, ny)]) * inv_dz2;
            }
        }
    }
}

static void wesi_zero_boundaries(float *field, int nx, int ny, int nz) {
    int x;
    int y;
    int z;
    for (z = 0; z < nz; ++z) {
        for (y = 0; y < ny; ++y) {
            field[wesi_index3(0, y, z, nx, ny)] = 0.0f;
            field[wesi_index3(nx - 1, y, z, nx, ny)] = 0.0f;
        }
    }
    for (z = 0; z < nz; ++z) {
        for (x = 0; x < nx; ++x) {
            field[wesi_index3(x, 0, z, nx, ny)] = 0.0f;
            field[wesi_index3(x, ny - 1, z, nx, ny)] = 0.0f;
        }
    }
    for (y = 0; y < ny; ++y) {
        for (x = 0; x < nx; ++x) {
            field[wesi_index3(x, y, 0, nx, ny)] = 0.0f;
            field[wesi_index3(x, y, nz - 1, nx, ny)] = 0.0f;
        }
    }
}

int wesi_run_forward(
    const wesi_submodel_t *submodel,
    const wesi_shot_t *shot,
    const wesi_horizon_set_t *horizons,
    const wesi_sim_params_t *params,
    wesi_checkpoint_t *checkpoint
) {
    (void)horizons;
    int status = wesi_validate(submodel, shot, params, checkpoint);
    if (status != 0) {
        return status;
    }

    int nx = submodel->grid.nx;
    int ny = submodel->grid.ny;
    int nz = submodel->grid.nz;
    int nt = params->nt;
    size_t nxyz = (size_t)nx * (size_t)ny * (size_t)nz;
    float *prev = (float *)calloc(nxyz, sizeof(float));
    float *curr = (float *)calloc(nxyz, sizeof(float));
    float *next = (float *)calloc(nxyz, sizeof(float));
    float *lap = (float *)calloc(nxyz, sizeof(float));
    if (prev == NULL || curr == NULL || next == NULL || lap == NULL) {
        free(prev);
        free(curr);
        free(next);
        free(lap);
        return -10;
    }

    float inv_dx2 = 1.0f / (submodel->grid.dx * submodel->grid.dx);
    float inv_dy2 = 1.0f / (submodel->grid.dy * submodel->grid.dy);
    float inv_dz2 = 1.0f / (submodel->grid.dz * submodel->grid.dz);
    size_t source_index = wesi_index3(shot->source_x, shot->source_y, shot->source_z, nx, ny);

    for (int it = 0; it < nt; ++it) {
        memset(next, 0, nxyz * sizeof(float));
        wesi_laplacian(curr, lap, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2);

        #pragma omp parallel for
        for (ptrdiff_t idx = 0; idx < (ptrdiff_t)nxyz; ++idx) {
            float coeff = submodel->velocity[idx] * params->dt;
            coeff = coeff * coeff;
            next[idx] = 2.0f * curr[idx] - prev[idx] + coeff * lap[idx];
        }
        next[source_index] += shot->wavelet[it];
        wesi_zero_boundaries(next, nx, ny, nz);

        for (int ir = 0; ir < shot->receiver_count; ++ir) {
            int rx = shot->receiver_xyz[3 * ir + 0];
            int ry = shot->receiver_xyz[3 * ir + 1];
            int rz = shot->receiver_xyz[3 * ir + 2];
            checkpoint->recorded_data[(size_t)it * (size_t)shot->receiver_count + (size_t)ir] = next[wesi_index3(rx, ry, rz, nx, ny)];
        }
        if (params->save_forward_wavefield && checkpoint->forward_wavefield != NULL) {
            memcpy(checkpoint->forward_wavefield + (size_t)it * nxyz, next, nxyz * sizeof(float));
        }

        memcpy(prev, curr, nxyz * sizeof(float));
        memcpy(curr, next, nxyz * sizeof(float));
    }

    free(prev);
    free(curr);
    free(next);
    free(lap);
    return 0;
}

int wesi_run_rtm(
    const wesi_submodel_t *submodel,
    const wesi_shot_t *shot,
    const wesi_horizon_set_t *horizons,
    const wesi_sim_params_t *params,
    wesi_checkpoint_t *checkpoint
) {
    (void)horizons;
    int status = wesi_validate(submodel, shot, params, checkpoint);
    if (status != 0) {
        return status;
    }
    if (checkpoint->image == NULL) {
        return -6;
    }
    if (shot->observed_data == NULL && checkpoint->recorded_data == NULL) {
        return -7;
    }

    int nx = submodel->grid.nx;
    int ny = submodel->grid.ny;
    int nz = submodel->grid.nz;
    int nt = params->nt;
    size_t nxyz = (size_t)nx * (size_t)ny * (size_t)nz;
    float *prev = (float *)calloc(nxyz, sizeof(float));
    float *curr = (float *)calloc(nxyz, sizeof(float));
    float *next = (float *)calloc(nxyz, sizeof(float));
    float *lap = (float *)calloc(nxyz, sizeof(float));
    if (prev == NULL || curr == NULL || next == NULL || lap == NULL) {
        free(prev);
        free(curr);
        free(next);
        free(lap);
        return -10;
    }

    memset(checkpoint->image, 0, nxyz * sizeof(float));
    float inv_dx2 = 1.0f / (submodel->grid.dx * submodel->grid.dx);
    float inv_dy2 = 1.0f / (submodel->grid.dy * submodel->grid.dy);
    float inv_dz2 = 1.0f / (submodel->grid.dz * submodel->grid.dz);
    const float *observed = shot->observed_data != NULL ? shot->observed_data : checkpoint->recorded_data;

    for (int it = nt - 1; it >= 0; --it) {
        memset(next, 0, nxyz * sizeof(float));
        wesi_laplacian(curr, lap, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2);

        #pragma omp parallel for
        for (ptrdiff_t idx = 0; idx < (ptrdiff_t)nxyz; ++idx) {
            float coeff = submodel->velocity[idx] * params->dt;
            coeff = coeff * coeff;
            next[idx] = 2.0f * curr[idx] - prev[idx] + coeff * lap[idx];
        }

        for (int ir = 0; ir < shot->receiver_count; ++ir) {
            int rx = shot->receiver_xyz[3 * ir + 0];
            int ry = shot->receiver_xyz[3 * ir + 1];
            int rz = shot->receiver_xyz[3 * ir + 2];
            next[wesi_index3(rx, ry, rz, nx, ny)] += observed[(size_t)it * (size_t)shot->receiver_count + (size_t)ir];
        }
        wesi_zero_boundaries(next, nx, ny, nz);

        if (checkpoint->forward_wavefield != NULL) {
            const float *forward = checkpoint->forward_wavefield + (size_t)it * nxyz;
            #pragma omp parallel for
            for (ptrdiff_t idx = 0; idx < (ptrdiff_t)nxyz; ++idx) {
                checkpoint->image[idx] += next[idx] * forward[idx];
            }
        }

        memcpy(prev, curr, nxyz * sizeof(float));
        memcpy(curr, next, nxyz * sizeof(float));
    }

    free(prev);
    free(curr);
    free(next);
    free(lap);
    return 0;
}
