# SolarGPU

This implementation is supplementary material for the paper *GPU-based solar irradiance estimation over Digital Surface Models using structurally lossless viewshed compression* by Niko Lukač and Borut Žalik, University of Maribor. 

**Version: 1.0**

## Description

GPU-accelerated annual solar irradiance estimation pipeline over a digital surface model (DSM), with shadowing, diffuse, direct and reflective irradiance. This implementation supports a compact "ba-RLE" compressed representation for the viewshed neighbor lists that the reflection pass streams through each time step.

Given a GeoTIFF of DSM heightmap, the pipeline:

1. Voxelizes it: one "top" voxel per pixel, plus stacked "wall" voxels.
2. On the GPU, out-of-core: traces a discretized hemisphere of directions  from every voxel to build per-voxel list of viewshed-neighbors.
3. For every hour of a full year: solar irradiance computation.

Steps 2-3 run **twice**, back-to-back, as two separate GPU pipeline
instances that differ only in how a voxel's neighbor list is stored:

- **`uncompressed_gpu_irradiance`**: a CSR-compact array (flat neighbor
  ids + a per-voxel offset table).
- **`compressed_gpu_irradiance`**: the same neighbor lists, ba-RLE
  compressed and decoded on the fly inside the reflection GPU kernel.

## Building

Requires a CUDA-capable GPU + toolkit (nvcc), CMake >= 3.18, and libtiff.

```bash
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..   # match your GPU's compute capability
cmake --build . -j
```

## Running

```bash
./SolarGPU <dsm.tiff> <lon> <lat> <tzone> <direct.txt> <diffuse.txt>
```

- `[dsm.tiff]`: DSM's heightmap (float32 format)
- `[lon]`: longitude
- `[lat]`: latitude
- `[tzone]`: timezone
- `[direct.txt]`: Hourly TMY time-serises of direct irradiance
- `[diffuse.txt]`: Hourly TMY time-serises of diffuse irradiance

## Outputs

- `<file>_solar_potential.tiff`: annual irradiance raster
- `<file>_voxels.bin`: per-voxel (grid_x, grid_y, z, annual_irradiance)
  point dump
