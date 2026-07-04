#pragma once

/* Shared out-of-core driver for stage 1 (hemispherical viewshed + SVF).
   Both GPU pipeline instances (compressed and uncompressed) call this with
   their own per-chunk consumer; only what happens to a chunk's raw neighbor
   ids after they're collected differs between them. */

#include <algorithm>
#include <cstdio>
#include <vector>

#include "cuda_utils.cuh"
#include "voxel_types.h"
#include "sky_bins.h"
#include "pipeline_math_constants.h"

#include "cuda/device_globals.cuh"
#include "cuda/gpu_memory_budget.cuh"
#include "cuda/kernels/viewshed_kernels.cuh"

namespace solar_gpu {

/* Traces the full hemisphere-pair of directions for every voxel, building
   the SVF bitmap (returned, fully resident on the device, the caller owns
   it and must cudaFree it) and streaming each voxel's neighbor-id list to
   `consume_chunk` in bounded-size chunks, so the raw (pre-trim) neighbor
   buffer never exceeds a safe fraction of free GPU memory. Each chunk is
   trimmed to its own local max neighbor count before being handed off --
   there's no single global "max neighbors" the way there would be with one
   big buffer, which both downstream storage formats (CSR offsets, ba-RLE
   streams) are already fine with.

   consume_chunk(voxel_offset, chunk_count, trimmed_max, host_raw_chunk) is
   called once per chunk with that chunk's trimmed, zero-padded raw neighbor
   ids (chunk_count * trimmed_max ints, in chunk-local voxel order). */
template <typename ConsumeChunk>
grid_s_type* run_viewshed_out_of_core(
    const float* grid_d, const int* pix_parent_d, const float* wall_z_d,
    int num_total, const SkyBins& bins, ConsumeChunk&& consume_chunk)
{
    size_t svf_bytes = (size_t)bins.hgrid_size_s * num_total * sizeof(grid_s_type);
    grid_s_type* grid_s_d = nullptr;
    cudaSafeCall(cudaMalloc((void**)&grid_s_d, svf_bytes));
    cudaSafeCall(cudaMemset(grid_s_d, 0, svf_bytes));

    const int init_max_neighbors = (int)(bins.h_hsize * bins.h_vsize_full);
    const int chunk_voxels = choose_viewshed_chunk_voxels(init_max_neighbors, num_total);
    std::printf("  [viewshed] chunk size: %d / %d voxels (%.1f MB/chunk, %d chunk%s)\n",
                chunk_voxels, num_total,
                (double)chunk_voxels * init_max_neighbors * sizeof(int) / 1048576.0,
                (num_total + chunk_voxels - 1) / chunk_voxels,
                (chunk_voxels < num_total) ? "s" : "");

    constexpr int kBlockSize = 256;

    for (int voxel_offset = 0; voxel_offset < num_total; voxel_offset += chunk_voxels) {
        const int chunk_count = std::min(chunk_voxels, num_total - voxel_offset);
        const int gridSizeChunk = (chunk_count + kBlockSize - 1) / kBlockSize;

        int* neighbor_ids_raw_d = nullptr;
        cudaSafeCall(cudaMalloc((void**)&neighbor_ids_raw_d, (size_t)chunk_count * init_max_neighbors * sizeof(int)));
        cudaSafeCall(cudaMemset(neighbor_ids_raw_d, 0, (size_t)chunk_count * init_max_neighbors * sizeof(int)));

        for (unsigned int i = 0; i < bins.h_hsize; ++i) {
            for (unsigned int j = 0; j < bins.h_vsize_full; ++j) {
                float3 Cdir; int is_upper; int dense_svf_idx = -1;
                if (j < bins.h_vsize) {
                    float e = (j + 0.5f) * bins.h_vres;
                    Cdir.x = cosf(e * kDegToRad) * sinf(((i + 0.5f) * bins.h_hres) * kDegToRad);
                    Cdir.y = -cosf(e * kDegToRad) * cosf(((i + 0.5f) * bins.h_hres) * kDegToRad);
                    Cdir.z = sinf(e * kDegToRad); is_upper = 1;
                    dense_svf_idx = bins.full_to_dense[i * bins.h_vsize + j];
                } else {
                    float e = (j - bins.h_vsize + 0.5f) * bins.h_vres;
                    Cdir.x = cosf(e * kDegToRad) * sinf(((i + 0.5f) * bins.h_hres) * kDegToRad);
                    Cdir.y = -cosf(e * kDegToRad) * cosf(((i + 0.5f) * bins.h_hres) * kDegToRad);
                    Cdir.z = -sinf(e * kDegToRad); is_upper = 0;
                }
                float len = sqrtf(Cdir.x * Cdir.x + Cdir.y * Cdir.y + Cdir.z * Cdir.z);
                if (len > 0.0f) { Cdir.x /= len; Cdir.y /= len; Cdir.z /= len; }

                calc_kernel_hemispherical_viewshed_GPU<<<gridSizeChunk, kBlockSize>>>(
                    Cdir, grid_d, pix_parent_d, wall_z_d, grid_s_d,
                    neighbor_ids_raw_d, init_max_neighbors, dense_svf_idx,
                    is_upper, voxel_offset, chunk_count);
                cudaCheckError(); cudaDeviceSynchronize();
            }
        }

        // trim this chunk to its own local max neighbor count
        int trimmed_max = init_max_neighbors;
        {
            int* bm_d = nullptr;
            cudaSafeCall(cudaMalloc((void**)&bm_d, (size_t)gridSizeChunk * sizeof(int)));
            count_max_neighbors_kernel<<<gridSizeChunk, kBlockSize, kBlockSize * sizeof(int)>>>(
                neighbor_ids_raw_d, init_max_neighbors, bm_d, chunk_count);
            cudaCheckError(); cudaDeviceSynchronize();

            std::vector<int> bm_h(gridSizeChunk);
            cudaSafeCall(cudaMemcpy(bm_h.data(), bm_d, (size_t)gridSizeChunk * sizeof(int), cudaMemcpyDeviceToHost));
            cudaSafeCall(cudaFree(bm_d));
            int gm = *std::max_element(bm_h.begin(), bm_h.end());
            trimmed_max = (gm > 0) ? gm : 1;

            if (trimmed_max < init_max_neighbors) {
                int* trimmed_d = nullptr;
                cudaSafeCall(cudaMalloc((void**)&trimmed_d, (size_t)chunk_count * trimmed_max * sizeof(int)));
                compact_neighbors_kernel<<<gridSizeChunk, kBlockSize>>>(
                    neighbor_ids_raw_d, trimmed_d, init_max_neighbors, trimmed_max, chunk_count);
                cudaCheckError(); cudaDeviceSynchronize();
                cudaSafeCall(cudaFree(neighbor_ids_raw_d));
                neighbor_ids_raw_d = trimmed_d;
            }
        }

        std::vector<int> chunk_host((size_t)chunk_count * trimmed_max);
        cudaSafeCall(cudaMemcpy(chunk_host.data(), neighbor_ids_raw_d,
                                 chunk_host.size() * sizeof(int), cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaFree(neighbor_ids_raw_d));

        consume_chunk(voxel_offset, chunk_count, trimmed_max, chunk_host);
    }

    return grid_s_d;
}

} // namespace solar_gpu
