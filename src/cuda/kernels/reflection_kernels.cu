#include "cuda/kernels/reflection_kernels.cuh"
#include "cuda/device_globals.cuh"
#include "barle_compression.h"
#include "pipeline_constants.h"

namespace {

/* Diffuse inter-reflection contribution from one neighbor voxel (identified
   by its linear grid index `ngid`) toward the receiving voxel described by
   (my_gx, my_gy, my_z) and surface normal (nvx, nvy, nvz). Shared by both
   neighbor-storage variants below - only how `ngid` is obtained differs.

   A neighbor found by the viewshed ray tracer is always a plain grid cell
   (never a wall voxel), so once it passes the in-bounds check below, its
   hourly irradiance sits at hourly_data[n_gy * grid_width + n_gx] directly
   - top voxels are exactly the grid's cells, in raster order, with no
   separate windowing/offset to account for. */
__device__ __forceinline__ void accumulate_reflection(
    int ngid, int my_gx, int my_gy, float my_z,
    float nvx, float nvy, float nvz,
    const float* __restrict__ grid, const float* __restrict__ hourly_data,
    float& R)
{
    if (ngid < 0 || ngid >= grid_width * grid_height) return;

    int n_gx = ngid % grid_width, n_gy = ngid / grid_width;
    float dx = (float)(n_gx - my_gx), dy = (float)(n_gy - my_gy), dz = grid[ngid] - my_z;
    float dist2 = dx * dx + dy * dy + dz * dz;
    if (dist2 < 1e-4f) return;
    float dist = sqrtf(dist2);

    float cos_recv = (nvx * dx + nvy * dy + nvz * dz) / dist;
    if (cos_recv <= 0.0f) return;

    float cos_send = fmaxf(fabsf(-dz) / dist, 0.1f);

    R += solar_gpu::kAlbedo * hourly_data[n_gy * grid_width + n_gx] * cos_recv * cos_send / (3.14159265f * dist2);
}

/* Common per-voxel setup: this voxel's grid position/height and the surface
   normal implied by its slope/aspect. Both reflection kernel variants need
   exactly this before walking their neighbor list. */
struct ReceiverVoxel {
    int gx, gy;
    float z;
    float nvx, nvy, nvz;
};

__device__ __forceinline__ ReceiverVoxel receiver_voxel(
    int t_id, const int* __restrict__ pix_parent, const float* __restrict__ wall_z,
    const float* __restrict__ grid, const float* __restrict__ slopes, const float* __restrict__ aspects)
{
    ReceiverVoxel v;
    int top_id = pix_parent[t_id];
    v.gx = top_id % grid_width;
    v.gy = top_id / grid_width;
    v.z = (t_id < num_top) ? grid[v.gx + v.gy * grid_width] : wall_z[t_id - num_top];

    float slope = slopes[t_id], aspect = aspects[t_id];
    float sin_s = sinf(slope);
    v.nvx = sin_s * sinf(aspect);
    v.nvy = -sin_s * cosf(aspect);
    v.nvz = cosf(slope);
    return v;
}

} // namespace

// ---- CSR-compact array ----
__global__ void calc_kernel_reflection_csr(
    const float* __restrict__ hourly_data, const float* __restrict__ grid,
    const float* __restrict__ slopes, const float* __restrict__ aspects,
    const int* __restrict__ pix_parent, const float* __restrict__ wall_z,
    const int* __restrict__ nb_flat, const int* __restrict__ nb_offsets,
    float* __restrict__ reflected_out, int total_pixels)
{
    const int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_id >= total_pixels) return;

    ReceiverVoxel me = receiver_voxel(t_id, pix_parent, wall_z, grid, slopes, aspects);

    float R = 0.0f;
    int start = nb_offsets[t_id];
    int end = nb_offsets[t_id + 1];
    for (int k = start; k < end; ++k) {
        accumulate_reflection(nb_flat[k] - 1, me.gx, me.gy, me.z, me.nvx, me.nvy, me.nvz, grid, hourly_data, R);
    }
    reflected_out[t_id] = R;
}

// ---- ba-RLE compressed stream ----

namespace {

__device__ __forceinline__ unsigned int read_bytes_be(const unsigned char* data, int& idx, int nBytes) {
    unsigned int v = 0;
    for (int i = 0; i < nBytes; ++i) { v <<= 8; v |= data[idx++]; }
    return v;
}

} // namespace

__global__ void calc_kernel_reflection_compressed(
    const float* __restrict__ hourly_data, const float* __restrict__ grid,
    const float* __restrict__ slopes, const float* __restrict__ aspects,
    const int* __restrict__ pix_parent, const float* __restrict__ wall_z,
    const unsigned char* __restrict__ comp_data, const int* __restrict__ comp_offsets,
    float* __restrict__ reflected_out, int total_pixels)
{
    const int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_id >= total_pixels) return;

    ReceiverVoxel me = receiver_voxel(t_id, pix_parent, wall_z, grid, slopes, aspects);

    float R = 0.0f;

    int start = comp_offsets[t_id];
    int end = comp_offsets[t_id + 1];
    const unsigned char* data = comp_data + start;
    int size = end - start;
    int idx = 0;

    unsigned int prev_val = 0;
    bool first = true;

    /* ba-RLE block stream: [3 bytes packed width codes][one length byte per
       run][delta payload] ... repeated until the per-voxel byte range
       [0,size) is consumed. Each packed-code field holds up to 8 three-bit
       width codes; solar_gpu::barle::kEndOfVoxel marks the first unused
       slot. See solar_gpu::barle in barle_compression.h for the encoder
       this must stay in lock-step with. */
    while (idx < size) {
        uint32_t bits = ((uint32_t)data[idx] << 16) |
            ((uint32_t)data[idx + 1] << 8) |
            ((uint32_t)data[idx + 2]);
        idx += 3;

        unsigned char widths[8];
        int nc = 0;
        for (int j = 0; j < 8; ++j) {
            unsigned char code = (unsigned char)((bits >> (21 - j * 3)) & 0x07);
            if (code == solar_gpu::barle::kEndOfVoxel) break;
            widths[nc++] = code;
        }

        unsigned char lens[8];
        for (int j = 0; j < nc; ++j) lens[j] = data[idx++];

        for (int j = 0; j < nc; ++j) {
            unsigned char width = widths[j];
            int runLen = (int)lens[j];

            if (width == solar_gpu::barle::kRunWidth4) {
                for (int i = 0; i < runLen; ) {
                    unsigned char packed = data[idx++];
                    unsigned int high = (packed >> 4) & 0x0F;
                    unsigned int val = first ? high : (prev_val - high);
                    first = false;
                    prev_val = val;
                    if (val > 0) accumulate_reflection((int)val - 1, me.gx, me.gy, me.z, me.nvx, me.nvy, me.nvz, grid, hourly_data, R);
                    ++i;
                    if (i < runLen) {
                        unsigned int low = packed & 0x0F;
                        val = prev_val - low;
                        prev_val = val;
                        if (val > 0) accumulate_reflection((int)val - 1, me.gx, me.gy, me.z, me.nvx, me.nvy, me.nvz, grid, hourly_data, R);
                        ++i;
                    }
                }
            } else {
                for (int i = 0; i < runLen; ++i) {
                    unsigned int delta = read_bytes_be(data, idx, (int)width);
                    unsigned int val = first ? delta : (prev_val - delta);
                    first = false;
                    prev_val = val;
                    if (val > 0) accumulate_reflection((int)val - 1, me.gx, me.gy, me.z, me.nvx, me.nvy, me.nvz, grid, hourly_data, R);
                }
            }
        }
    }

    reflected_out[t_id] = R;
}
