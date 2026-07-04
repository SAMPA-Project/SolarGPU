#pragma once

/* __constant__ / __device__ symbols shared by every kernel translation
   unit, plus the host-side helper that uploads them. Symbols are defined
   exactly once, in src/cuda/device_globals.cu; every .cu file that uses
   them includes this header and relies on CUDA relocatable device code to
   resolve the symbol across translation units (see CMakeLists.txt:
   CUDA_SEPARABLE_COMPILATION ON).

   There's no separate "target tile within a larger mosaic" concept in this
   version of the pipeline, the single loaded heightmap *is* the target,
   so a voxel's top-pixel index converts to (x, y) directly via
   grid_width, with no offset. */

#include "pipeline_constants.h"
#include "sky_bins.h"

extern __constant__ int grid_width;
extern __constant__ int grid_height;

extern __constant__ unsigned int num_data; // = num_total voxels (top + wall)
extern __constant__ int          num_top;

// ---- hemisphere direction-bin grid used for the SVF bitmap ----
extern __constant__ unsigned int hsize, vsize; // bin counts, horizontal/vertical
extern __constant__ float        hres;         // horizontal bin size, degrees
extern __constant__ float        vres;         // vertical bin size, degrees

extern __constant__ float maxz; // max heightmap value; bounds upward viewshed rays

extern __constant__ float F_matrix[6][8];

extern __device__ float2 sun_azm_alt[solar_gpu::kHoursPerYear];    // (azimuth, altitude), degrees
extern __device__ int    sun_dense_idx_d[solar_gpu::kHoursPerYear]; // dense SVF bin index for that hour's sun direction, -1 if none
extern __device__ float  direct_ir[solar_gpu::kHoursPerYear];
extern __device__ float  diffuse_ir[solar_gpu::kHoursPerYear];

namespace solar_gpu {

/* Uploads every one of the above symbols in one call. Defined in
   device_globals.cu (the same translation unit the symbols themselves are
   defined in), shared by both GPU pipeline instances. */
void upload_pipeline_constants(int h_grid_width, int h_grid_height, int h_num_top, int h_num_total,
                                float h_max_height, const SkyBins& bins,
                                const std::vector<float>& direct_irradiance,
                                const std::vector<float>& diffuse_irradiance);

} // namespace solar_gpu
