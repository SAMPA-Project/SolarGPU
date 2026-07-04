#include "sky_bins.h"

#include <cmath>

#include "voxel_types.h"

namespace solar_gpu {

SkyBins compute_sky_bins(const std::vector<SunSample>& sun_positions, float h_res, float v_res) {
    SkyBins bins;
    bins.h_hres = h_res;
    bins.h_vres = v_res;
    bins.h_hsize = (unsigned int)std::ceil(360.0f / h_res);
    bins.h_vsize = (unsigned int)std::ceil(90.0f / v_res);
    bins.h_vsize_full = bins.h_vsize * 2;

    /* Tiny epsilon nudge keeps a sun position sitting exactly on a bin
       boundary from rounding up into the next bin. */
    bins.sun_azm_alt.resize(kHoursPerYear);
    std::vector<bool> bin_used(bins.h_hsize * bins.h_vsize, false);
    for (int k = 0; k < kHoursPerYear; ++k) {
        float az = std::get<0>(sun_positions[k]) - 1e-8f;
        float alt = std::get<1>(sun_positions[k]) - 1e-8f;
        bins.sun_azm_alt[k] = SunSample(az, alt);
        if (alt < 1.0f) continue;

        unsigned int bi = (unsigned int)(az / h_res);
        unsigned int bj = (unsigned int)(alt / v_res);
        if (bi < bins.h_hsize && bj < bins.h_vsize) bin_used[bi * bins.h_vsize + bj] = true;
    }

    bins.full_to_dense.assign((size_t)bins.h_hsize * bins.h_vsize, -1);
    for (unsigned int i = 0; i < bins.h_hsize; ++i)
        for (unsigned int j = 0; j < bins.h_vsize; ++j)
            if (bin_used[i * bins.h_vsize + j])
                bins.full_to_dense[i * bins.h_vsize + j] = (int)bins.active_bin_count++;

    bins.hgrid_size_s = (unsigned int)std::ceil((float)bins.active_bin_count / (float)kGridSBits);

    bins.sun_dense_idx.resize(kHoursPerYear);
    for (int k = 0; k < kHoursPerYear; ++k) {
        float az = std::get<0>(bins.sun_azm_alt[k]);
        float alt = std::get<1>(bins.sun_azm_alt[k]);
        if (alt < 1.0f) { bins.sun_dense_idx[k] = -1; continue; }

        unsigned int bi = (unsigned int)(az / h_res);
        unsigned int bj = (unsigned int)(alt / v_res);
        bins.sun_dense_idx[k] = (bi < bins.h_hsize && bj < bins.h_vsize) ? bins.full_to_dense[bi * bins.h_vsize + bj] : -1;
    }

    return bins;
}

} // namespace solar_gpu
