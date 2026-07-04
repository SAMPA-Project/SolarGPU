#pragma once

/* Loads a single GeoTIFF DSM raster as a flat height grid. There is no
   mosaicking in this version of the pipeline, one heightmap file is the
   entire world the shadowing/irradiance pipeline sees. */

#include <string>
#include <vector>

#include "geotiff_io.h"

namespace solar_gpu {

struct HeightGrid {
    int width = 0, height = 0;
    double scaleX = 0.0, scaleY = 0.0; // pixel size (map units), from the GeoTIFF pixel-scale tag
    std::vector<float> heights;        // width * height, row-major, north-up
};

/* Reads `path` as a single-band 32-bit float GeoTIFF and returns its height
   data plus pixel scale. `meta` is filled in so the caller can write
   georeferenced outputs back out with write_tiff(). Exits the process if
   the file is missing the pixel-scale tag needed to know the voxel size. */
HeightGrid load_heightmap(const std::string& path, TiffMeta& meta);

} // namespace solar_gpu
