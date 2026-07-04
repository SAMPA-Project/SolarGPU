#pragma once

/* Stage 3 of the pipeline: for every voxel, sum diffusely-reflected
   irradiance arriving from its viewshed neighbors (the candidates found in
   stage 1) that are themselves lit this hour. The physical calculation is
   identical in both kernels below; they differ only in how the neighbor-id
   list is stored and walked. */

/* ---- CSR-compact array (the "uncompressed" pipeline instance) ----
   One flat array plus a per-voxel offset table (nb_offsets[i]..nb_offsets[i+1]);
   no padding, but reads across voxels aren't aligned to a fixed stride. */
__global__ void calc_kernel_reflection_csr(
    const float* __restrict__ hourly_data, const float* __restrict__ grid,
    const float* __restrict__ slopes, const float* __restrict__ aspects,
    const int* __restrict__ pix_parent, const float* __restrict__ wall_z,
    const int* __restrict__ nb_flat, const int* __restrict__ nb_offsets,
    float* __restrict__ reflected_out, int total_pixels);

/* ---- ba-RLE compressed stream (the "compressed" pipeline instance) ----
   Smallest memory footprint; each thread streams-decodes its own voxel's
   ba-RLE byte range on the fly (see solar_gpu::barle in barle_compression.h
   for the encoding this must stay in lock-step with). */
__global__ void calc_kernel_reflection_compressed(
    const float* __restrict__ hourly_data, const float* __restrict__ grid,
    const float* __restrict__ slopes, const float* __restrict__ aspects,
    const int* __restrict__ pix_parent, const float* __restrict__ wall_z,
    const unsigned char* __restrict__ comp_data, const int* __restrict__ comp_offsets,
    float* __restrict__ reflected_out, int total_pixels);
