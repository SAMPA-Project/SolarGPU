#pragma once

/* Physical/temporal constants shared across the pipeline. There are no more
   compile-time backend choices here (see README.md: both the compressed and
   uncompressed neighbor-storage pipelines are always built and always run). */

namespace solar_gpu {

constexpr float kAlbedo = 0.2f;          // diffuse ground/wall reflectance used by the reflection kernel
constexpr int   kHoursPerYear = 8760;    // non-leap-year hourly resolution for the TMY irradiance series

} // namespace solar_gpu
