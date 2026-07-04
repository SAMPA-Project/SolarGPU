#pragma once

/* Host-facing entry points into the CUDA pipeline. Two separate instances
   are always built and are meant to be run back-to-back on the same
   inputs, see src/cuda/gpu_irradiance_uncompressed.cu and
   gpu_irradiance_compressed.cu. This header has no CUDA dependency so it
   can be included from plain C++ translation units like src/main.cpp. */

#include <vector>

#include "sun_position.h"

namespace solar_gpu {

/* Runs shadowing + hourly direct/diffuse/reflected solar irradiance
   accumulation over a full year for every voxel (top + wall) of a single
   heightmap.

     grid_data         heightmap, grid_width * grid_height floats, row-major
     num_total_voxels / num_top_voxels
                        total voxel count (top + wall) and how many are top
                        pixels (== grid_width * grid_height); wall voxels are
                        indices [num_top_voxels, num_total_voxels)
     pixel_parent       per-voxel index of the top pixel it belongs to (size num_total_voxels)
     wall_z             per-wall-voxel height (size num_total_voxels - num_top_voxels)
     max_height         max height anywhere in the grid (viewshed ray termination bound)
     sun_positions      (azimuth_deg, altitude_deg) for every hour of the year (kHoursPerYear entries)
     direct_irradiance,
     diffuse_irradiance TMY direct/diffuse irradiance, W/m^2, one value per hour (kHoursPerYear entries)
     slopes, aspects    per-voxel surface slope/aspect, radians (size num_total_voxels)
     irradiance_out     [out] accumulated annual irradiance per voxel, W*h/m^2 (size num_total_voxels)

   Returns 0 on success.

   uncompressed_gpu_irradiance() stores each voxel's inter-reflection
   neighbor list as a CSR-compact array (flat ids + per-voxel offsets). */
int uncompressed_gpu_irradiance(const float* grid_data, int grid_width, int grid_height,
    int num_total_voxels, int num_top_voxels,
    const int* pixel_parent, const float* wall_z, float max_height,
    const std::vector<SunSample>& sun_positions,
    const std::vector<float>& direct_irradiance, const std::vector<float>& diffuse_irradiance,
    const float* slopes, const float* aspects, float* irradiance_out);

/* Same pipeline and identical inputs/outputs, but the inter-reflection
   neighbor lists are ba-RLE compressed and streamed/decoded on the fly
   inside the reflection kernel (see barle_compression.h). Smaller GPU
   memory footprint for the reflection pass; same physical result as
   uncompressed_gpu_irradiance() up to floating point rounding. */
int compressed_gpu_irradiance(const float* grid_data, int grid_width, int grid_height,
    int num_total_voxels, int num_top_voxels,
    const int* pixel_parent, const float* wall_z, float max_height,
    const std::vector<SunSample>& sun_positions,
    const std::vector<float>& direct_irradiance, const std::vector<float>& diffuse_irradiance,
    const float* slopes, const float* aspects, float* irradiance_out);

} // namespace solar_gpu
