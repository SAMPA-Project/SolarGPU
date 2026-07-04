cmake_minimum_required(VERSION 3.18)
project(SolarGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  # Turing/Ampere/Ada by default -- override for your GPU, e.g.:
  #   cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

find_package(TIFF REQUIRED)

add_executable(SolarGPU
  src/main.cpp
  src/geotiff_io.cpp
  src/heightmap.cpp
  src/voxelization.cpp
  src/sky_bins.cpp
  src/barle_compression.cpp
  src/cuda/device_globals.cu
  src/cuda/gpu_irradiance_uncompressed.cu
  src/cuda/gpu_irradiance_compressed.cu
  src/cuda/kernels/viewshed_kernels.cu
  src/cuda/kernels/solar_kernel.cu
  src/cuda/kernels/reflection_kernels.cu
)

target_include_directories(SolarGPU PRIVATE include)
target_link_libraries(SolarGPU PRIVATE TIFF::TIFF)

# The two pipeline instances (gpu_irradiance_uncompressed.cu,
# gpu_irradiance_compressed.cu) share kernels, __constant__/__device__
# globals, and the out-of-core viewshed stage across several .cu files
set_target_properties(SolarGPU PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
