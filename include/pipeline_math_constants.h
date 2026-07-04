#pragma once

/* Small shared constants used by both host (.cpp) and device (.cu)
   translation units. Centralized here so nobody has to guess whether M_PI
   is already defined by <cmath> / a system header on a given platform. */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace solar_gpu {

/* Degrees -> radians, float precision (matches the original SPA-derived
   sun-position pipeline, which works entirely in float). */
constexpr float kDegToRad = 0.0174532925f;

} // namespace solar_gpu
