/* Compressed pipeline instance: inter-reflection neighbor lists are ba-RLE
   compressed as each out-of-core viewshed chunk comes in, and streamed/
   decoded on the fly inside the reflection kernel. See
   gpu_irradiance_uncompressed.cu for the CSR-compact counterpart, both
   share the same kernels and out-of-core viewshed stage, and are meant to
   be run back-to-back on identical inputs (see main.cpp). */

#include "solar_gpu_api.h"

#include <chrono>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "voxel_types.h"
#include "sky_bins.h"
#include "barle_compression.h"

#include "cuda/device_globals.cuh"
#include "cuda/gpu_timing.cuh"
#include "cuda/viewshed_stage.cuh"
#include "cuda/kernels/solar_kernel.cuh"
#include "cuda/kernels/reflection_kernels.cuh"

namespace solar_gpu {

int compressed_gpu_irradiance(const float* grid_data, int h_grid_width, int h_grid_height,
    int num_total, int num_top_voxels,
    const int* pixel_parent, const float* wall_z_host, float max_height,
    const std::vector<SunSample>& sun_positions,
    const std::vector<float>& direct_irradiance, const std::vector<float>& diffuse_irradiance,
    const float* slopes, const float* aspects, float* irradiance_out)
{
    auto t_total = std::chrono::high_resolution_clock::now();

    cudaDeviceProp prop; cudaSetDevice(0); cudaGetDeviceProperties(&prop, 0);
    std::cout << "[compressed] Using GPU: " << prop.name << "\n";
    float initialmem = print_gpu_mem("initial");

    cudaFuncSetCacheConfig(calc_kernel_hemispherical_viewshed_GPU, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(calc_kernel_solar, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(calc_kernel_reflection_compressed, cudaFuncCachePreferL1);

    const int num_grid_cells = h_grid_width * h_grid_height;
    const int num_walls = num_total - num_top_voxels;

    SkyBins bins = compute_sky_bins(sun_positions);
    std::cout << "[*] Sun visits " << bins.active_bin_count << " / " << (bins.h_hsize * bins.h_vsize)
        << " upper hemisphere bins (" << (100.0f * bins.active_bin_count / (bins.h_hsize * bins.h_vsize)) << "%)\n";

    upload_pipeline_constants(h_grid_width, h_grid_height, num_top_voxels, num_total, max_height, bins,
                               direct_irradiance, diffuse_irradiance);

    // ---- device allocations ----
    float* grid_d = nullptr; float* slopes_d = nullptr; float* aspects_d = nullptr;
    int* pix_parent_d = nullptr; float* wall_z_d = nullptr;
    float* irradiance_d = nullptr; float* hourly_d = nullptr; float* reflected_d = nullptr;

    cudaSafeCall(cudaMalloc((void**)&grid_d, (size_t)num_grid_cells * sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)&slopes_d, (size_t)num_total * sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)&aspects_d, (size_t)num_total * sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)&pix_parent_d, (size_t)num_total * sizeof(int)));
    if (num_walls > 0) {
        cudaSafeCall(cudaMalloc((void**)&wall_z_d, (size_t)num_walls * sizeof(float)));
        cudaSafeCall(cudaMemcpy(wall_z_d, wall_z_host, (size_t)num_walls * sizeof(float), cudaMemcpyHostToDevice));
    }
    cudaSafeCall(cudaMalloc((void**)&irradiance_d, (size_t)num_total * sizeof(float)));
    cudaSafeCall(cudaMemset(irradiance_d, 0, (size_t)num_total * sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)&hourly_d, (size_t)num_total * sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)&reflected_d, (size_t)num_total * sizeof(float)));

    cudaSafeCall(cudaMemcpy(grid_d, grid_data, (size_t)num_grid_cells * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(slopes_d, slopes, (size_t)num_total * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(aspects_d, aspects, (size_t)num_total * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(pix_parent_d, pixel_parent, (size_t)num_total * sizeof(int), cudaMemcpyHostToDevice));

    // ---- [1] out-of-core viewshed -> ba-RLE compressed neighbor storage ----
    std::cout << "[1] Computing viewshed + neighbors (ba-RLE compressed)\n";
    auto t_stage1 = std::chrono::high_resolution_clock::now();

    std::vector<int> comp_offsets(num_total + 1, 0);
    std::vector<barle::Byte> comp_flat;
    size_t total_raw_bytes = 0;

    grid_s_type* grid_s_d = run_viewshed_out_of_core(
        grid_d, pix_parent_d, wall_z_d, num_total, bins,
        [&](int voxel_offset, int chunk_count, int trimmed_max, const std::vector<int>& chunk_raw) {
            std::vector<int> chunk_offsets;
            std::vector<barle::Byte> chunk_flat;
            size_t chunk_raw_bytes = 0;
            barle::compress_all_neighbors(chunk_raw.data(), chunk_count, trimmed_max,
                                           chunk_offsets, chunk_flat, chunk_raw_bytes);

            int base_offset = (int)comp_flat.size();
            for (int i = 0; i < chunk_count; ++i)
                comp_offsets[voxel_offset + i] = base_offset + chunk_offsets[i];
            comp_flat.insert(comp_flat.end(), chunk_flat.begin(), chunk_flat.end());
            total_raw_bytes += chunk_raw_bytes;
        });
    comp_offsets[num_total] = (int)comp_flat.size();

    std::cout << std::fixed << "  Compression ratio: " << comp_flat.size() / 1024.0 << " KB from "
        << total_raw_bytes / 1024.0 << " KB ("
        << (100.0 * comp_flat.size() / std::max<size_t>(total_raw_bytes, 1)) << "%)\n";
    std::cout << "  [TIME] Viewshed + compression: " << elapsed_ms(t_stage1) << " ms\n";

    unsigned char* comp_data_d = nullptr; int* comp_offsets_d = nullptr;
    cudaSafeCall(cudaMalloc((void**)&comp_offsets_d, (size_t)(num_total + 1) * sizeof(int)));
    cudaSafeCall(cudaMemcpy(comp_offsets_d, comp_offsets.data(), (size_t)(num_total + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (!comp_flat.empty()) {
        cudaSafeCall(cudaMalloc((void**)&comp_data_d, comp_flat.size()));
        cudaSafeCall(cudaMemcpy(comp_data_d, comp_flat.data(), comp_flat.size(), cudaMemcpyHostToDevice));
    }
    float finalmem = print_gpu_mem("after neighbor upload");

    // ---- [2] solar + reflection, hourly over the year ----
    std::cout << "[2] Solar irradiance + reflection.\n";
    auto t_stage2 = std::chrono::high_resolution_clock::now();
    constexpr int blockSize1D = 256;
    const int gridSize1D = (num_total + blockSize1D - 1) / blockSize1D;
    int hours_computed = 0;

    for (int day = 0; day < 365; ++day) {
        for (int hour = 0; hour < 24; ++hour) {
            int idx = day * 24 + hour;
            if (std::get<1>(bins.sun_azm_alt[idx]) < 1.0f) continue;

            cudaSafeCall(cudaMemset(hourly_d, 0, (size_t)num_total * sizeof(float)));
            cudaSafeCall(cudaMemset(reflected_d, 0, (size_t)num_total * sizeof(float)));

            calc_kernel_solar<<<gridSize1D, blockSize1D>>>(grid_s_d, hourly_d, slopes_d, aspects_d, day, hour, num_total);
            cudaCheckError(); cudaDeviceSynchronize();

            calc_kernel_reflection_compressed<<<gridSize1D, blockSize1D>>>(
                hourly_d, grid_d, slopes_d, aspects_d, pix_parent_d, wall_z_d,
                comp_data_d, comp_offsets_d, reflected_d, num_total);
            cudaCheckError(); cudaDeviceSynchronize();

            accumulate_kernel<<<gridSize1D, blockSize1D>>>(irradiance_d, hourly_d, reflected_d, num_total);
            cudaCheckError(); cudaDeviceSynchronize();
            ++hours_computed;
        }
    }
    std::cout << "  [TIME] Solar+reflection (" << hours_computed << " hrs): " << elapsed_ms(t_stage2) << " ms\n";

    cudaSafeCall(cudaMemcpy(irradiance_out, irradiance_d, (size_t)num_total * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- cleanup ----
    cudaSafeCall(cudaFree(grid_d));
    cudaSafeCall(cudaFree(slopes_d));
    cudaSafeCall(cudaFree(aspects_d));
    cudaSafeCall(cudaFree(pix_parent_d));
    if (wall_z_d) cudaSafeCall(cudaFree(wall_z_d));
    cudaSafeCall(cudaFree(grid_s_d));
    cudaSafeCall(cudaFree(irradiance_d));
    cudaSafeCall(cudaFree(hourly_d));
    cudaSafeCall(cudaFree(reflected_d));
    if (comp_data_d) cudaSafeCall(cudaFree(comp_data_d));
    cudaSafeCall(cudaFree(comp_offsets_d));

    std::cout << "  [TIME] Total (compressed): " << elapsed_ms(t_total) << " ms\n";
    std::printf("  [GPU MEM] diff: %7.1f MB \n", finalmem - initialmem);

    return 0;
}

} // namespace solar_gpu
