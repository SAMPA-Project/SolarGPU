#include "voxelization.h"

#include <algorithm>
#include <cmath>

#include "pipeline_math_constants.h"

namespace solar_gpu {

std::vector<WallVoxel> detect_wall_voxels(const HeightGrid& grid, float voxelSize) {
    std::vector<WallVoxel> walls;

    /* Sentinel returned for any neighbour that falls outside the grid.
       Anything above -kOpenOutside/2 is treated as "open air" by the column
       depth calculation below; kept far from any real elevation value. */
    constexpr float kOpenOutside = -1e30f;

    auto height_at = [&](int xx, int yy) -> float {
        if (xx >= 0 && xx < grid.width && yy >= 0 && yy < grid.height)
            return grid.heights[(size_t)yy * grid.width + xx];
        return kOpenOutside;
    };

    for (int gy = 0; gy < grid.height; ++gy) {
        for (int gx = 0; gx < grid.width; ++gx) {
            float z0 = grid.heights[(size_t)gy * grid.width + gx];
            int topIdx = gy * grid.width + gx;

            float hE = height_at(gx + 1, gy), hW = height_at(gx - 1, gy);
            float hN = height_at(gx, gy - 1), hS = height_at(gx, gy + 1);

            /* Column depth only uses real neighbour heights - clamp the
               out-of-grid sentinel to z0 so an open side doesn't drag the
               apparent drop to -infinity. */
            float clampedE = (hE <= kOpenOutside * 0.5f) ? z0 : hE;
            float clampedW = (hW <= kOpenOutside * 0.5f) ? z0 : hW;
            float clampedN = (hN <= kOpenOutside * 0.5f) ? z0 : hN;
            float clampedS = (hS <= kOpenOutside * 0.5f) ? z0 : hS;
            float minNeighbor = std::min({ clampedE, clampedW, clampedN, clampedS });

            float drop = z0 - minNeighbor;
            if (drop <= voxelSize) continue;

            int numVoxels = (int)(drop / voxelSize);
            for (int v = 0; v < numVoxels; ++v) {
                float zz = z0 - (v + 0.5f) * voxelSize;

                // Openness on each side at this height 
                float openE = std::max(0.0f, zz - hE);
                float openW = std::max(0.0f, zz - hW);
                float openN = std::max(0.0f, zz - hN);
                float openS = std::max(0.0f, zz - hS);

                float vx = openE - openW, vyNorth = openN - openS;
                float aspect = std::atan2(vx, vyNorth);
                if (aspect < 0.0f) aspect += 2.0f * (float)M_PI;

                walls.push_back({ topIdx, zz, aspect });
            }
        }
    }

    return walls;
}

SlopeAspect compute_top_slope_aspect(const HeightGrid& grid) {
    SlopeAspect out;
    const size_t numTop = (size_t)grid.width * grid.height;
    out.slope.resize(numTop);
    out.aspect.resize(numTop);

    for (int gy = 0; gy < grid.height; ++gy) {
        for (int gx = 0; gx < grid.width; ++gx) {
            int idx = gy * grid.width + gx;

            int gxL = std::max(gx - 1, 0);
            int gxR = std::min(gx + 1, grid.width - 1);
            int gyU = std::max(gy - 1, 0);
            int gyD = std::min(gy + 1, grid.height - 1);

            float zL = grid.heights[(size_t)gy * grid.width + gxL];
            float zR = grid.heights[(size_t)gy * grid.width + gxR];
            float zU = grid.heights[(size_t)gyU * grid.width + gx];
            float zD = grid.heights[(size_t)gyD * grid.width + gx];

            float dx = (zR - zL) / ((gxR - gxL) * (float)grid.scaleX);
            float dy = (zD - zU) / ((gyD - gyU) * (float)grid.scaleY);

            out.slope[idx] = std::atan2(std::sqrt(dx * dx + dy * dy), 1.0f);
            float aspect = std::atan2(-dx, dy);
            if (aspect < 0.0f) aspect += 2.0f * (float)M_PI;
            out.aspect[idx] = aspect;
        }
    }

    return out;
}

} // namespace solar_gpu
