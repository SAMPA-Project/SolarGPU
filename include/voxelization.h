#pragma once

#include <vector>

#include "heightmap.h"

namespace solar_gpu {

struct WallVoxel {
    int parentIdx;  // index of the top pixel this wall column hangs from
    float z;        // voxel center height, map units
    float aspect;   // outward-facing normal azimuth, radians; 0 = north, clockwise positive
};

/* Column-drop wall detection: for every pixel in `grid`, compare its height
   against its 4 planar (N/S/E/W) neighbours. */
std::vector<WallVoxel> detect_wall_voxels(const HeightGrid& grid, float voxelSize);

/* Per-top-pixel slope/aspect via central differences (edge pixels clamp to
   the nearest interior neighbour rather than wrapping or reading out of
   bounds). */
struct SlopeAspect {
    std::vector<float> slope;   // radians from vertical (0 = flat)
    std::vector<float> aspect;  // radians; 0 = north, clockwise positive
};

SlopeAspect compute_top_slope_aspect(const HeightGrid& grid);

} // namespace solar_gpu
