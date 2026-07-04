#pragma once

/* Picks a safe voxel-chunk size for the out-of-core viewshed stage (see
   viewshed_stage.cuh) based on currently-free GPU memory, so the raw
   (pre-trim, one-slot-per-possible-direction) neighbor buffer never grows
   large enough to exhaust the device, this is the buffer that would
   otherwise dominate GPU memory use even before ba-RLE compression ever
   gets a chance to help. */

#include <algorithm>
#include <cstddef>

#include <cuda_runtime.h>

namespace solar_gpu {

/* Returns how many voxels' worth of a `max_neighbors`-wide raw neighbor
   buffer can be held on the GPU at once, out of `total_voxels`. */
inline int choose_viewshed_chunk_voxels(int max_neighbors, int total_voxels) {
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    /* Reserve headroom: the trim/compact step briefly needs a second
       same-sized (or smaller) buffer alongside the first, and other
       pipeline buffers (heightmap, slopes, SVF bitmap, ...) are already
       resident by the time this runs. */
    constexpr double kUsableFraction = 0.4;
    size_t usable_bytes = (size_t)((double)free_bytes * kUsableFraction);

    size_t bytes_per_voxel = (size_t)max_neighbors * sizeof(int);
    if (bytes_per_voxel == 0) return total_voxels;

    size_t chunk = std::min(usable_bytes / bytes_per_voxel, (size_t)total_voxels);
    return (int)std::max<size_t>(chunk, 1);
}

} // namespace solar_gpu
