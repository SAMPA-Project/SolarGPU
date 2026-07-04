/* Uncompressed pipeline instance: inter-reflection neighbor lists are
   stored as a CSR-compact array (flat ids + per-voxel offset table). See
   gpu_irradiance_compressed.cu for the ba-RLE-compressed counterpart,
   both share the same kernels and out-of-core viewshed stage, and are
   meant to be run back-to-back on identical inputs (see main.cpp). */

#include "solar_gpu_api.h"

#include <chrono>
#include <iostream>
#include <vector>

#include "cuda_utils.cuh"
#include "voxel_types.h"
#include "sky_bins.h"

#include "cuda/device_globals.cuh"
#include "cuda/gpu_timing.cuh"
#include "cuda/viewshed_stage.cuh"
#include "cuda/kernels/solar_kernel.cuh"
#include "cuda/kernels/reflection_kernels.cuh"

namespace solar_gpu {

int uncompressed_gpu_irradiance(const float* grid_data, int h_grid_width, int h_grid_height,
    int num_total, int num_top_voxels,
    const int* pixel_parent, const float* wall_z_host, float max_height,
    const std::vector<SunSample>& sun_positions,
    const std::vector<float>& direct_irradiance, const std::vector<float>& diffuse_irradiance,
    const float* slopes, const float* aspects, float* irradiance_out)
{
    auto t_total = std::chrono::high_resolution_clock::now();

    cudaDeviceProp prop; cudaSetDevice(0); cudaGetDeviceProperties(&prop, 0);
    std::cout << "[uncompressed] Using GPU: " << prop.name << "\n";
    float initialmem = print_gpu_mem("initial");

    cudaFuncSetCacheConfig(calc_kernel_hemispherical_viewshed_GPU, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(calc_kernel_solar, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(calc_kernel_reflection_csr, cudaFuncCachePreferL1);

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

    // ---- [1] out-of-core viewshed -> CSR-compact neighbor storage ----
    std::cout << "[1] Computing viewshed + neighbors (CSR)\n";
    auto t_stage1 = std::chrono::high_resolution_clock::now();

    std::vector<int> nb_offsets(num_total + 1, 0);
    std::vector<int> nb_flat;

    grid_s_type* grid_s_d = run_viewshed_out_of_core(
        grid_d, pix_parent_d, wall_z_d, num_total, bins,
        [&](int voxel_offset, int chunk_count, int trimmed_max, const std::vector<int>& chunk_raw) {
            for (int i = 0; i < chunk_count; ++i) {
                nb_offsets[voxel_offset + i] = (int)nb_flat.size();
                size_t base = (size_t)i * trimmed_max;
                for (int k = 0; k < trimmed_max; ++k) {
                    int v = chunk_raw[base + k];
                    if (v == 0) break;
                    nb_flat.push_back(v);
                }
            }
        });
    nb_offsets[num_total] = (int)nb_flat.size();
    std::cout << "  [TIME] Viewshed: " << elapsed_ms(t_stage1) << " ms (" << nb_flat.size() << " total neighbors)\n";

    int* nb_offsets_d = nullptr; int* nb_flat_d = nullptr;
    cudaSafeCall(cudaMalloc((void**)&nb_offsets_d, (size_t)(num_total + 1) * sizeof(int)));
    cudaSafeCall(cudaMemcpy(nb_offsets_d, nb_offsets.data(), (size_t)(num_total + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (!nb_flat.empty()) {
        cudaSafeCall(cudaMalloc((void**)&nb_flat_d, nb_flat.size() * sizeof(int)));
        cudaSafeCall(cudaMemcpy(nb_flat_d, nb_flat.data(), nb_flat.size() * sizeof(int), cudaMemcpyHostToDevice));
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

            calc_kernel_reflection_csr<<<gridSize1D, blockSize1D>>>(
                hourly_d, grid_d, slopes_d, aspects_d, pix_parent_d, wall_z_d,
                nb_flat_d, nb_offsets_d, reflected_d, num_total);
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
    if (nb_flat_d) cudaSafeCall(cudaFree(nb_flat_d));
    cudaSafeCall(cudaFree(nb_offsets_d));

    std::cout << "  [TIME] Total (uncompressed): " << elapsed_ms(t_total) << " ms\n";
    std::printf("  [GPU MEM] diff: %7.1f MB \n", finalmem - initialmem);

    return 0;
}

} // namespace solar_gpu
