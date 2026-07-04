#pragma once

/* Stage 1 of the pipeline: for every voxel, trace a ray in every direction
   of a discretized hemisphere-pair (upper + lower) to (a) find candidate
   inter-reflection neighbors in every direction, and (b) mark which
   sun-visited upper-hemisphere directions are unobstructed, building the
   sky-view-factor (SVF) bitmap consumed by the solar kernel.

   Runs out-of-core over voxel chunks (see viewshed_stage.cuh) so the raw
   neighbor-id buffer, by far the largest allocation in the pipeline --
   never has to cover every voxel at once. */

#include "voxel_types.h"

/* Traces one ray direction (Cdir) from every voxel in
   [voxel_offset, voxel_offset + chunk_count) outward using a 2D DDA over the
   heightmap (the voxel's origin is offset 2 grid cells along Cdir first, to
   avoid immediately self-intersecting its own column) and records the first
   taller cell it hits as a neighbor candidate.

   `pix_parent`, `wall_z`, and `grid_s` are indexed by *global* voxel id
   (voxel_offset + local thread id) since they cover every voxel in the
   grid; `neighbor_ids` is indexed by *local* thread id since it's a
   chunk-sized scratch buffer, freed and reallocated for the next chunk.

   dense_svf_idx >= 0: this direction is a sun-visited upper-hemisphere bin;
                       set the corresponding SVF bit if the ray is unobstructed.
   dense_svf_idx <  0: lower hemisphere, or an upper bin the sun never visits
                       this year, neighbor collection still happens, but no
                       SVF bit is touched. */
__global__ void calc_kernel_hemispherical_viewshed_GPU(
    const float3 Cdir, const float* __restrict__ grid,
    const int* __restrict__ pix_parent, const float* __restrict__ wall_z,
    solar_gpu::grid_s_type* __restrict__ grid_s,
    int* __restrict__ neighbor_ids, int max_neighbors,
    int dense_svf_idx, int is_upper,
    int voxel_offset, int chunk_count);

/* Block-wise reduction over a chunk's neighbor buffer: for each voxel
   counts its (zero-terminated) number of collected neighbors, then reduces
   to the max count per block. The host takes the max over all blocks to
   find that chunk's true max neighbor count, letting its (very
   conservative) padded buffer be trimmed before it's downloaded and folded
   into the final storage representation. */
__global__ void count_max_neighbors_kernel(
    const int* __restrict__ neighbor_ids, int max_neighbors,
    int* __restrict__ block_max, int chunk_count);

/* Copies each voxel's first `new_max` neighbor slots from a wider chunk
   buffer into a narrower one (the trim step described above). */
__global__ void compact_neighbors_kernel(
    const int* __restrict__ src, int* __restrict__ dst,
    int old_max, int new_max, int chunk_count);
