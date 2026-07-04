#pragma once

/* Host-side "ba-RLE" compression for a voxel's viewshed neighbor-id list
   Decoder lives in src/cuda/kernels/reflection_kernels.cu and must be kept
   in lock-step with the encoding here if this ever changes.

   The scheme, per voxel:
     1. Collect that voxel's neighbor ids (grid indices + 1; 0 = unused slot).
     2. Sort descending, dedupe, delta-encode consecutive values.
     3. Classify each delta by the minimum byte width (4-bit/8/16/24/32) that
        holds it, and run-length-encode consecutive same-width deltas.
     4. Optimize the run list: absorb single narrower outliers into a wider
        neighboring run when that's cheaper than a separate run header, then
        coalesce adjacent same-width runs.
     5. Pack into blocks of up to 8 runs: 3 bytes of packed 3-bit width codes,
        then one length byte per run, then each run's delta payload
        (4-bit deltas packed two per byte; wider ones as big-endian bytes).

   A block's code field is padded with kEndOfVoxel (7) once a voxel's runs
   run out, which is how the GPU decoder knows where one voxel's byte range
   ends without needing a separate length table entry per block. */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace solar_gpu::barle {

using Byte = std::uint8_t;

/* Per-run neighbor-delta byte widths. Values fit in the 3-bit "code" field
   packed into each block header. */
enum RunWidth : Byte {
    kRunWidth4 = 0,  // deltas 0-15, two packed per byte
    kRunWidth8 = 1,  // deltas 0-255
    kRunWidth16 = 2,
    kRunWidth24 = 3,
    kRunWidth32 = 4,
};

/* Sentinel written into a block's 3-bit code field once a voxel's runs run
   out. Must not collide with the RunWidth values (0..4); the remaining
   3-bit values 5/6/7 are all free, 7 is used for clarity. */
constexpr Byte kEndOfVoxel = 7;

/* Compresses one voxel's neighbor id list.

   `raw` holds up to `max_neighbors` ids, zero-terminated (grid cell index +
   1; 0 marks the end of the list, since 0 isn't a valid 1-based id).
   `out_raw_bytes` receives the equivalent uncompressed size (id count * 4)
   for computing compression ratios. */
std::vector<Byte> compress_voxel_neighbors(const int* raw, int max_neighbors, int& out_raw_bytes);

/* Compresses every voxel's neighbor list and concatenates the results.
   offsets[i]..offsets[i+1] bounds voxel i's byte range in flat_data
   (offsets has num_total + 1 entries). out_raw_bytes accumulates the
   uncompressed-equivalent size across all voxels. */
void compress_all_neighbors(const int* raw_host, int num_total, int max_neighbors,
                             std::vector<int>& offsets, std::vector<Byte>& flat_data,
                             std::size_t& out_raw_bytes);

} // namespace solar_gpu::barle
