/* solar-irradiance-gpu CLI: given one GeoTIFF heightmap and a TMY direct/
   diffuse irradiance series, computes annual per-voxel solar irradiance
   (with shadowing and diffuse inter-reflection) and writes the result back
   out as GeoTIFFs. Runs both GPU pipeline instances: CSR-compact and
   ba-RLE compressed neighbor storage, back-to-back on the same inputs, so
   their timing, memory use, and results can be compared directly. */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "geotiff_io.h"
#include "heightmap.h"
#include "voxelization.h"
#include "sun_position.h"
#include "solar_gpu_api.h"
#include "pipeline_constants.h"
#include "pipeline_math_constants.h"

namespace fs = std::filesystem;
using namespace solar_gpu;

namespace {

std::vector<float> read_tmy_series(const std::string& path, size_t n) {
    std::vector<float> v;
    v.reserve(n);
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open TMY irradiance file: " << path << "\n";
        std::exit(EXIT_FAILURE);
    }
    std::copy_n(std::istream_iterator<float>(ifs), n, std::back_inserter(v));
    return v;
}

/* Dumps every voxel's (grid_x, grid_y, z, annual_irradiance) as raw floats,
   mainly for point-cloud style visual inspection of a run's output. */
void write_voxel_dump(const std::string& path, int num_total, int num_top, int grid_width,
                       const std::vector<int>& pixel_parent, const std::vector<float>& grid_heights,
                       const std::vector<float>& wall_z, const std::vector<float>& irradiance) {
    std::ofstream vf(path, std::ios::binary);
    if (!vf.is_open()) {
        std::cerr << "Warning: cannot write " << path << "\n";
        return;
    }

    vf.write((const char*)&num_total, sizeof(int));
    vf.write((const char*)&num_top, sizeof(int));
    for (int i = 0; i < num_total; ++i) {
        int top_id = pixel_parent[i];
        int gx = top_id % grid_width;
        int gy = top_id / grid_width;
        float x = (float)gx, y = (float)gy;
        float z = (i < num_top) ? grid_heights[(size_t)gy * grid_width + gx] : wall_z[i - num_top];
        float val = irradiance[i];
        vf.write((const char*)&x, sizeof(float));
        vf.write((const char*)&y, sizeof(float));
        vf.write((const char*)&z, sizeof(float));
        vf.write((const char*)&val, sizeof(float));
    }
}

} // namespace

int main(int argc, char* argv[]) {
    register_geotiff_tags();

    const std::string heightmap_path = (argc > 1) ? argv[1] : "test_data/dsm_heightmap.tiff";
    float lon = (argc > 2) ? (float)std::atof(argv[2]) : 15.592; 
    float lat = (argc > 3) ? (float)std::atof(argv[3]) : 46.543; 
    /* UTC offset the TMY series' hour-of-day index is anchored to (fixed local
       standard time, no DST). Must match how tmy_direct.txt/tmy_diffuse.txt
       were indexed, or sun position and measured irradiance desync
       hour-for-hour. */
    int timezone = (argc > 4) ? std::atoi(argv[4]) : (int)std::lround(lon / 15.0f);

    // TMY hourly direct/diffuse irradiance (W/m^2)
    const std::string direct_irrad_path =  (argc > 5) ? argv[5] : "test_data/tmy_direct.txt";
    const std::string diffuse_irrad_path =  (argc > 6) ? argv[6] : "test_data/tmy_diffuse.txt";

    auto direct_irrad = read_tmy_series(direct_irrad_path, kHoursPerYear);
    auto diffuse_irrad = read_tmy_series(diffuse_irrad_path, kHoursPerYear);

    // ---- load the heightmap ----
    TiffMeta meta;
    HeightGrid grid = load_heightmap(heightmap_path, meta);

    const int numTop = grid.width * grid.height;
    const float host_maxz = *std::max_element(grid.heights.begin(), grid.heights.end());

    // ---- sun positions for the whole year ----
    std::vector<SunSample> sunAzAl;
    SunPositionCalculator sun(lon, lat, timezone);
    sun.calc(sunAzAl);

    // ---- voxelize ----
    const float voxelSize = (float)grid.scaleX;
    std::vector<WallVoxel> walls = detect_wall_voxels(grid, voxelSize);
    SlopeAspect topSlopeAspect = compute_top_slope_aspect(grid);

    const int numWalls = (int)walls.size();
    const int numTotal = numTop + numWalls;
    std::cout << "Top voxels: " << numTop << ", wall voxels: " << numWalls
        << ", total voxels: " << numTotal << "\n";

    std::vector<int> pixel_parent(numTotal);
    std::vector<float> all_slopes(numTotal), all_aspects(numTotal);
    std::vector<float> wall_z(numWalls > 0 ? numWalls : 0);

    for (int idx = 0; idx < numTop; ++idx) {
        pixel_parent[idx] = idx;
        all_slopes[idx] = topSlopeAspect.slope[idx];
        all_aspects[idx] = topSlopeAspect.aspect[idx];
    }
    for (int i = 0; i < numWalls; ++i) {
        int idx = numTop + i;
        pixel_parent[idx] = walls[i].parentIdx;
        all_slopes[idx] = (float)M_PI * 0.5f;
        all_aspects[idx] = walls[i].aspect;
        wall_z[i] = walls[i].z;
    }

    // ---- shadowing + reflection on the GPU: both pipeline instances ----
    std::vector<float> irradiance_uncompressed(numTotal);
    std::vector<float> irradiance_compressed(numTotal);

    std::cout << "\n=== Uncompressed (CSR) pipeline ===\n";
    uncompressed_gpu_irradiance(grid.heights.data(), grid.width, grid.height, numTotal, numTop,
        pixel_parent.data(), wall_z.empty() ? nullptr : wall_z.data(), host_maxz,
        sunAzAl, direct_irrad, diffuse_irrad,
        all_slopes.data(), all_aspects.data(), irradiance_uncompressed.data());

    std::cout << "\n=== Compressed (ba-RLE) pipeline ===\n";
    compressed_gpu_irradiance(grid.heights.data(), grid.width, grid.height, numTotal, numTop,
        pixel_parent.data(), wall_z.empty() ? nullptr : wall_z.data(), host_maxz,
        sunAzAl, direct_irrad, diffuse_irrad,
        all_slopes.data(), all_aspects.data(), irradiance_compressed.data());

    // ---- quick sanity check: the two pipelines should agree closely ----
    double max_abs_diff = 0.0, sum_abs_diff = 0.0;
    //double max_irrad = 0.0;
    //int max_irrad_idx = -1;
    for (int i = 0; i < numTotal; ++i) {
        double d = std::fabs((double)irradiance_uncompressed[i] - (double)irradiance_compressed[i]);
        max_abs_diff = std::max(max_abs_diff, d);
        /*if ((double)irradiance_uncompressed[i] > max_irrad) {
            max_irrad = (double)irradiance_uncompressed[i];
            max_irrad_idx = i;
        }*/
        sum_abs_diff += d;
    }
    std::cout << "\nCompressed vs uncompressed: max abs diff = " << max_abs_diff
        << ", mean abs diff = " << (sum_abs_diff / numTotal) << "\n";

    /*
    std::cout << "Max. annual irradiance (MWh): " << max_irrad / 1000000.0 << "\n";
    if (max_irrad_idx >= 0) {
        int top_id = pixel_parent[max_irrad_idx];
        int gx = top_id % grid.width, gy = top_id / grid.width;
        std::cout << "  DEBUG max voxel idx=" << max_irrad_idx
            << " (top_id=" << top_id << ", gx=" << gx << ", gy=" << gy << ")"
            << " is_wall=" << (max_irrad_idx >= numTop)
            << " slope_deg=" << (all_slopes[max_irrad_idx] * 180.0 / M_PI)
            << " aspect_deg=" << (all_aspects[max_irrad_idx] * 180.0 / M_PI)
            << " z=" << grid.heights[(size_t)gy * grid.width + gx] << "\n";
    }
    */

    // ---- write outputs (already in grid.width x grid.height raster order, matching meta) ----
    const std::string outName = fs::path(heightmap_path).filename().string();

    write_tiff(outName + "_solar_potential.tiff", irradiance_compressed, meta);

    write_voxel_dump(outName + "_voxels.bin", numTotal, numTop, grid.width, pixel_parent, grid.heights, wall_z, irradiance_compressed);

    return 0;
}
