#include "geotiff_io.h"

#include <algorithm>
#include <cstring>
#include <iostream>

#include <tiffio.h>

namespace solar_gpu {

namespace {

/* libtiff's classic API doesn't know about GeoTIFF tags out of the box; this
   table teaches it their type/count so TIFFGetField/TIFFSetField work for
   them. See the GeoTIFF spec for tag numbers: 33550 ModelPixelScaleTag,
   33922 ModelTiepointTag, 34735 GeoKeyDirectoryTag, 34736 GeoDoubleParamsTag,
   34737 GeoAsciiParamsTag. 42113 (GDAL_NODATA) is GDAL's convention, not part
   of the GeoTIFF spec proper, but tiles produced by GDAL commonly carry it. */
const TIFFFieldInfo kGeotiffFieldInfo[] = {
    { 33550, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"ModelPixelScaleTag" },
    { 33922, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"ModelTiepointTag" },
    { 34735, -1, -1, TIFF_SHORT,  FIELD_CUSTOM, 1, 1, (char*)"GeoKeyDirectoryTag" },
    { 34736, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"GeoDoubleParamsTag" },
    { 34737, -1, -1, TIFF_ASCII,  FIELD_CUSTOM, 1, 0, (char*)"GeoAsciiParamsTag" },
    { 42113, -1, -1, TIFF_ASCII,  FIELD_CUSTOM, 1, 0, (char*)"GDAL_NODATA" },
};

TIFFExtendProc g_parentExtender = nullptr;

void geotiff_tag_extender(TIFF* tif) {
    TIFFMergeFieldInfo(tif, kGeotiffFieldInfo, sizeof(kGeotiffFieldInfo) / sizeof(kGeotiffFieldInfo[0]));
    if (g_parentExtender) g_parentExtender(tif);
}

[[noreturn]] void fail(const std::string& msg) {
    std::cerr << msg << "\n";
    std::exit(EXIT_FAILURE);
}

} // namespace

void register_geotiff_tags() {
    static bool registered = false;
    if (!registered) {
        g_parentExtender = TIFFSetTagExtender(geotiff_tag_extender);
        registered = true;
    }
}

std::vector<float> read_tiff(const std::string& path, TiffMeta& meta) {
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (!tif) fail("Cannot open " + path);

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &meta.w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &meta.h);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &meta.sampleFormat);
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &meta.bitsPerSample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &meta.samplesPerPixel);
    TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &meta.compression);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PHOTOMETRIC, &meta.photometric);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &meta.planarConfig);
    if (!TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &meta.rowsPerStrip))
        meta.rowsPerStrip = meta.h;

    /* GeoTIFF tags: libtiff hands back pointers into its own internal
       buffers, so copy everything out into owned vectors immediately. */
    {
        double* tiepoints = nullptr; uint16_t tiepointCount = 0;
        double* pixelScale = nullptr; uint16_t pixelScaleCount = 0;
        uint16_t* geoKeys = nullptr; uint16_t geoKeyCount = 0;
        double* geoDoubles = nullptr; uint16_t geoDoubleCount = 0;
        char* geoAscii = nullptr; uint16_t geoAsciiCount = 0;

        TIFFGetField(tif, 33922, &tiepointCount, &tiepoints);
        TIFFGetField(tif, 33550, &pixelScaleCount, &pixelScale);
        TIFFGetField(tif, 34735, &geoKeyCount, &geoKeys);
        TIFFGetField(tif, 34736, &geoDoubleCount, &geoDoubles);
        TIFFGetField(tif, 34737, &geoAsciiCount, &geoAscii);

        if (tiepoints && tiepointCount)
            meta.tiepoints.assign(tiepoints, tiepoints + tiepointCount);
        if (pixelScale && pixelScaleCount)
            meta.pixelScale.assign(pixelScale, pixelScale + pixelScaleCount);
        if (geoKeys && geoKeyCount)
            meta.geoKeys.assign(geoKeys, geoKeys + geoKeyCount);
        if (geoDoubles && geoDoubleCount)
            meta.geoDoubles.assign(geoDoubles, geoDoubles + geoDoubleCount);
        if (geoAscii && geoAsciiCount)
            meta.geoAscii.assign(geoAscii, geoAscii + geoAsciiCount);
    }

    if (meta.bitsPerSample != 32 || meta.sampleFormat != SAMPLEFORMAT_IEEEFP) {
        TIFFClose(tif);
        fail("Expected 32-bit float TIFF, got bps=" + std::to_string(meta.bitsPerSample) +
             " fmt=" + std::to_string(meta.sampleFormat) + " (" + path + ")");
    }

    std::vector<float> data((size_t)meta.w * meta.h);

    if (TIFFIsTiled(tif)) {
        uint32_t tileW = 0, tileH = 0;
        if (!TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileW) ||
            !TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH)) {
            TIFFClose(tif);
            fail("Tiled TIFF missing tile dimensions: " + path);
        }

        tsize_t tileBytes = TIFFTileSize(tif);
        float* tileBuf = (float*)_TIFFmalloc(tileBytes);
        if (!tileBuf) {
            TIFFClose(tif);
            fail("Out of memory for tile buffer");
        }

        for (uint32_t ty = 0; ty < meta.h; ty += tileH) {
            for (uint32_t tx = 0; tx < meta.w; tx += tileW) {
                if (TIFFReadTile(tif, tileBuf, tx, ty, 0, 0) < 0) {
                    _TIFFfree(tileBuf);
                    TIFFClose(tif);
                    fail("Error reading tile at (" + std::to_string(tx) + ", " + std::to_string(ty) + ")");
                }
                uint32_t copyW = std::min<uint32_t>(tileW, meta.w - tx);
                uint32_t copyH = std::min<uint32_t>(tileH, meta.h - ty);
                for (uint32_t r = 0; r < copyH; ++r) {
                    std::memcpy(data.data() + (size_t)(ty + r) * meta.w + tx,
                        tileBuf + (size_t)r * tileW,
                        copyW * sizeof(float));
                }
            }
        }
        _TIFFfree(tileBuf);
    } else {
        for (uint32_t row = 0; row < meta.h; ++row) {
            if (TIFFReadScanline(tif, data.data() + (size_t)row * meta.w, row) < 0) {
                TIFFClose(tif);
                fail("Error reading row " + std::to_string(row) + " of " + path);
            }
        }
    }

    TIFFClose(tif);
    return data;
}

void write_tiff(const std::string& path, const std::vector<float>& data, const TiffMeta& meta) {
    TIFF* tif = TIFFOpen(path.c_str(), "w");
    if (!tif) fail("Cannot create " + path);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, meta.w);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, meta.h);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, meta.compression);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, meta.rowsPerStrip);

    if (!meta.tiepoints.empty())
        TIFFSetField(tif, 33922, (uint16_t)meta.tiepoints.size(), meta.tiepoints.data());
    if (!meta.pixelScale.empty())
        TIFFSetField(tif, 33550, (uint16_t)meta.pixelScale.size(), meta.pixelScale.data());
    if (!meta.geoKeys.empty())
        TIFFSetField(tif, 34735, (uint16_t)meta.geoKeys.size(), meta.geoKeys.data());
    if (!meta.geoDoubles.empty())
        TIFFSetField(tif, 34736, (uint16_t)meta.geoDoubles.size(), meta.geoDoubles.data());
    if (!meta.geoAscii.empty())
        TIFFSetField(tif, 34737, (uint16_t)meta.geoAscii.size(), meta.geoAscii.c_str());

    for (uint32_t row = 0; row < meta.h; ++row) {
        if (TIFFWriteScanline(tif, (void*)(data.data() + (size_t)row * meta.w), row) < 0) {
            TIFFClose(tif);
            fail("Error writing row " + std::to_string(row) + " of " + path);
        }
    }

    TIFFClose(tif);
}

} // namespace solar_gpu
