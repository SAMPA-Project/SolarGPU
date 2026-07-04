#pragma once

/* Types shared between host (.cpp) and device (.cu) code for the per-voxel
   sky-view-factor (SVF) bitmap. Usable from both plain C++ and nvcc
   translation units, the __host__ __device__ qualifiers below only apply
   under nvcc (see SGPU_HD). */

#include <cstddef>

#ifdef __CUDACC__
#define SGPU_HD __host__ __device__
#else
#define SGPU_HD
#endif

namespace solar_gpu {

/* One SVF bit per sun-visited upper-hemisphere direction bin, per voxel.
   grid_s_type is the storage word; kGridSBits bits are packed per word. */
using grid_s_type = unsigned char;

constexpr unsigned int kGridSBits = sizeof(grid_s_type) * 8u;

/* Layout is word-major / voxel-minor: for a given direction bin's word index,
   all voxels' words are contiguous. This keeps the per-direction viewshed
   kernel's SVF-bit writes coalesced across voxels (each kernel launch only
   ever touches one bit position, shared by every thread). */
SGPU_HD inline size_t svf_word_index(size_t bit_loc) {
    return bit_loc / kGridSBits;
}

SGPU_HD inline unsigned svf_bit_index(size_t bit_loc) {
    return static_cast<unsigned>(bit_loc % kGridSBits);
}

SGPU_HD inline size_t svf_word_offset(size_t voxel_id, size_t bit_loc, size_t num_voxels) {
    return voxel_id + svf_word_index(bit_loc) * num_voxels;
}

SGPU_HD inline grid_s_type get_svf_bit(size_t voxel_id, size_t bit_loc,
                                        const grid_s_type* grid_s, size_t num_voxels) {
    return (grid_s[svf_word_offset(voxel_id, bit_loc, num_voxels)] >> svf_bit_index(bit_loc))
           & grid_s_type(1);
}

SGPU_HD inline void set_svf_bit(size_t voxel_id, size_t bit_loc,
                                 grid_s_type* grid_s, size_t num_voxels) {
    grid_s[svf_word_offset(voxel_id, bit_loc, num_voxels)] |=
        grid_s_type(grid_s_type(1) << svf_bit_index(bit_loc));
}

} // namespace solar_gpu
