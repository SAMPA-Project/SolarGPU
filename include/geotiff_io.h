#pragma once

// Reading and writing single-band, 32-bit float GeoTIFF rasters 

#include <cstdint>
#include <string>
#include <vector>

namespace solar_gpu {

/* GeoTIFF metadata for one raster: standard TIFF image tags plus the
   GeoTIFF-specific tags (tiepoint/pixel-scale/geo-keys) needed to place the
   tile in map space and to write a georeferenced output raster. */
struct TiffMeta {
    uint32_t w = 0, h = 0;
    uint16_t sampleFormat = 0, bitsPerSample = 0, samplesPerPixel = 0;
    uint16_t compression = 0, photometric = 0, planarConfig = 0;
    uint32_t rowsPerStrip = 0;

    std::vector<double> tiepoints;   // GeoTIFF tag 33922 (ModelTiepointTag)
    std::vector<double> pixelScale;  // GeoTIFF tag 33550 (ModelPixelScaleTag)
    std::vector<uint16_t> geoKeys;   // GeoTIFF tag 34735 (GeoKeyDirectoryTag)
    std::vector<double> geoDoubles;  // GeoTIFF tag 34736 (GeoDoubleParamsTag)
    std::string geoAscii;            // GeoTIFF tag 34737 (GeoAsciiParamsTag)
};

/* Registers the GeoTIFF custom tags (33550/33922/34735/34736/34737) plus
   GDAL_NODATA (42113) with libtiff. Must be called once before any
   read_tiff()/write_tiff() call. Safe to call more than once. */
void register_geotiff_tags();

/* Reads a single-band, 32-bit IEEE float TIFF (tiled or scanline, either is
   handled transparently). Exits the process with an error message if the
   file can't be opened, is missing required tags, or isn't 32-bit float --
   this mirrors a CLI tool's fail-fast behaviour rather than a library
   convention, so callers embedding this in a larger application will likely
   want to replace the exit() calls in geotiff_io.cpp with exceptions. */
std::vector<float> read_tiff(const std::string& path, TiffMeta& meta);

/* Writes `data` (meta.w * meta.h floats, row-major) as a single-band 32-bit
   float GeoTIFF, carrying over meta's GeoTIFF tags so the output lands in
   the same coordinate reference system as the input. */
void write_tiff(const std::string& path, const std::vector<float>& data, const TiffMeta& meta);

} // namespace solar_gpu
