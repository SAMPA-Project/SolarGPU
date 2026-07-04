#pragma once

/* Precomputed sun-visited-bin bookkeeping, shared by both GPU pipeline
   instances (compressed and uncompressed). Pure host C++, no CUDA
   dependency. */

#include <vector>

#include "pipeline_constants.h"
#include "sun_position.h"

namespace solar_gpu {

/* Which upper-hemisphere direction bins the sun ever occupies over the
   year, and a dense (gap-free) index for each of them, this is what keeps
   the SVF bitmap sized to only the bins that matter instead of the full
   hemisphere grid. */
struct SkyBins {
    unsigned int h_hsize = 0, h_vsize = 0, h_vsize_full = 0; // bin counts: horizontal, vertical (upper), vertical (upper+lower)
    float h_hres = 5.0f, h_vres = 5.0f;                       // bin size, degrees
    unsigned int active_bin_count = 0;
    unsigned int hgrid_size_s = 0;                            // SVF bitmap words per voxel

    std::vector<int> full_to_dense;    // size h_hsize*h_vsize; -1 if the sun never visits that bin
    std::vector<int> sun_dense_idx;    // size kHoursPerYear; -1 if the sun is below the horizon that hour
    std::vector<SunSample> sun_azm_alt; // size kHoursPerYear; epsilon-nudged copy of the input sun positions
};

/* `sun_positions` must have exactly kHoursPerYear entries (see
   SunPositionCalculator::calc()). */
SkyBins compute_sky_bins(const std::vector<SunSample>& sun_positions,
                          float h_res = 5.0f, float v_res = 5.0f);

} // namespace solar_gpu
