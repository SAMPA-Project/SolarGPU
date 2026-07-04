#include "heightmap.h"

#include <iostream>

namespace solar_gpu {

HeightGrid load_heightmap(const std::string& path, TiffMeta& meta) {
    HeightGrid grid;
    grid.heights = read_tiff(path, meta);

    if (meta.pixelScale.size() < 2) {
        std::cerr << "TIFF " << path << " missing GeoTIFF pixel-scale tag\n";
        std::exit(EXIT_FAILURE);
    }

    grid.width = (int)meta.w;
    grid.height = (int)meta.h;
    grid.scaleX = meta.pixelScale[0];
    grid.scaleY = meta.pixelScale[1];

    std::cout << "Heightmap: " << path << " (" << grid.width << "x" << grid.height << ")"
               << " scale=(" << grid.scaleX << ", " << grid.scaleY << ")\n";
    return grid;
}

} // namespace solar_gpu
