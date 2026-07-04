#pragma once

/* Stage 2 of the pipeline: per-voxel, per-hour direct + diffuse irradiance.
   Direct is gated by the SVF bit for the sun's current direction bin (i.e.
   "is this voxel's surface unobstructed toward the sun this hour") and by a
   [5deg, 89deg] solar-altitude guard band; diffuse uses a simple isotropic
   sky model weighted by surface tilt. Does not include inter-reflection --
   see reflection_kernels.cuh for that. */

#include "voxel_types.h"

__device__ inline float Perez(const float& cos_z_angle, const float& cos_angle_of_incidence, const float& dif_irad, const float& dir_irad, const float& slope, const int& n);

__global__ void calc_kernel_solar(solar_gpu::grid_s_type* grid_s, float* hourly_data,
    float* slopes, float* aspects, int day_of_year, int hour, int total_pixels);

/* total[i] += hourly[i] + reflected[i]. Called once per simulated hour to
   build up the annual sum in-place. */
__global__ void accumulate_kernel(float* __restrict__ total,
    const float* __restrict__ hourly, const float* __restrict__ reflected,
    int total_pixels);
