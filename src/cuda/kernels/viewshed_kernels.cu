#include "cuda/kernels/viewshed_kernels.cuh"
#include "cuda/device_globals.cuh"

using solar_gpu::grid_s_type;
using solar_gpu::set_svf_bit;

__global__ void calc_kernel_hemispherical_viewshed_GPU(
    const float3 Cdir, const float* __restrict__ grid,
    const int* __restrict__ pix_parent, const float* __restrict__ wall_z,
    grid_s_type* __restrict__ grid_s,
    int* __restrict__ neighbor_ids, int max_neighbors,
    const int dense_svf_idx,
    int is_upper, int voxel_offset, int chunk_count)
{
    const int local_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_id >= chunk_count) return;
    const int t_id = voxel_offset + local_id; // global voxel id

    int top_id = pix_parent[t_id];
    int gx = top_id % grid_width;
    int gy = top_id / grid_width;

    float base_z;
    if (t_id < num_top) base_z = grid[gx + gy * grid_width];
    else base_z = wall_z[t_id - num_top];

    const float ox = (float)gx + Cdir.x * 2.0f;
    const float oy = (float)gy + Cdir.y * 2.0f;
    const float oz = base_z + Cdir.z * 2.0f;
    int ix = __float2int_rd(ox);
    int iy = __float2int_rd(oy);

    float tMaxX = ((Cdir.x >= 0.0f) ? (ix + 1.0f - ox) : (ox - (float)ix)) / fmaxf(fabsf(Cdir.x), 1e-8f);
    float tMaxY = ((Cdir.y >= 0.0f) ? (iy + 1.0f - oy) : (oy - (float)iy)) / fmaxf(fabsf(Cdir.y), 1e-8f);
    float t = 0.0f;

    while (1) {
        if (tMaxX < tMaxY) { t = tMaxX; ix += (Cdir.x >= 0.0f) ? 1 : -1; tMaxX += 1.0f / fmaxf(fabsf(Cdir.x), 1e-8f); }
        else { t = tMaxY; iy += (Cdir.y >= 0.0f) ? 1 : -1; tMaxY += 1.0f / fmaxf(fabsf(Cdir.y), 1e-8f); }

        if ((unsigned)ix >= grid_width || (unsigned)iy >= grid_height) break;
        if (is_upper && (oz + Cdir.z * t > maxz)) break;
        if (!is_upper && (oz + Cdir.z * t < 0.0f)) break;

        if (grid[ix + iy * grid_width] - (oz + Cdir.z * t) > 0.01f) {
            if (dense_svf_idx >= 0)
                set_svf_bit((size_t)t_id, (size_t)dense_svf_idx, grid_s, num_data);

            int hit_val = ix + iy * grid_width + 1;
            size_t base = (size_t)local_id * max_neighbors; // chunk-local buffer
            for (int k = 0; k < max_neighbors; ++k) {
                int existing = neighbor_ids[base + k];
                if (existing == hit_val) break;
                if (existing == 0) { neighbor_ids[base + k] = hit_val; break; }
            }
            break;
        }
    }
}

__global__ void count_max_neighbors_kernel(
    const int* __restrict__ neighbor_ids, int max_neighbors,
    int* __restrict__ block_max, int chunk_count)
{
    const int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int cnt = 0;
    if (t_id < chunk_count) {
        size_t base = (size_t)t_id * max_neighbors;
        for (int k = 0; k < max_neighbors; ++k) { if (neighbor_ids[base + k] == 0) break; ++cnt; }
    }
    extern __shared__ int sdata[];
    sdata[threadIdx.x] = cnt;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && sdata[threadIdx.x + s] > sdata[threadIdx.x]) sdata[threadIdx.x] = sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) block_max[blockIdx.x] = sdata[0];
}

__global__ void compact_neighbors_kernel(
    const int* __restrict__ src, int* __restrict__ dst,
    int old_max, int new_max, int chunk_count)
{
    const int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_id >= chunk_count) return;
    size_t sb = (size_t)t_id * old_max, db = (size_t)t_id * new_max;
    for (int k = 0; k < new_max; ++k) dst[db + k] = src[sb + k];
}
