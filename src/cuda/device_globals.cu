#include "cuda/device_globals.cuh"

#include "cuda_utils.cuh"

// ---------------- shared device globals (declared extern in device_globals.cuh) ----------------

__constant__ int grid_width;
__constant__ int grid_height;

__constant__ unsigned int num_data;
__constant__ int          num_top;
__constant__ unsigned int hsize, vsize;
__constant__ float        hres;
__constant__ float        vres;
__constant__ float        maxz;
__constant__ float F_matrix[6][8];

__device__ float2 sun_azm_alt[solar_gpu::kHoursPerYear];
__device__ int    sun_dense_idx_d[solar_gpu::kHoursPerYear];
__device__ float  direct_ir[solar_gpu::kHoursPerYear];
__device__ float  diffuse_ir[solar_gpu::kHoursPerYear];

namespace solar_gpu {

void upload_pipeline_constants(int h_grid_width, int h_grid_height, int h_num_top, int h_num_total,
                                float h_max_height, const SkyBins& bins,
                                const std::vector<float>& direct_irradiance,
                                const std::vector<float>& diffuse_irradiance) {
    unsigned int h_num_data = (unsigned int)h_num_total;

    cudaSafeCall(cudaMemcpyToSymbol(grid_width, &h_grid_width, sizeof(int)));
    cudaSafeCall(cudaMemcpyToSymbol(grid_height, &h_grid_height, sizeof(int)));
    cudaSafeCall(cudaMemcpyToSymbol(num_top, &h_num_top, sizeof(int)));
    cudaSafeCall(cudaMemcpyToSymbol(num_data, &h_num_data, sizeof(unsigned int)));
    cudaSafeCall(cudaMemcpyToSymbol(hsize, &bins.h_hsize, sizeof(unsigned int)));
    cudaSafeCall(cudaMemcpyToSymbol(vsize, &bins.h_vsize, sizeof(unsigned int)));
    cudaSafeCall(cudaMemcpyToSymbol(hres, &bins.h_hres, sizeof(float)));
    cudaSafeCall(cudaMemcpyToSymbol(vres, &bins.h_vres, sizeof(float)));
    cudaSafeCall(cudaMemcpyToSymbol(maxz, &h_max_height, sizeof(float)));

    std::vector<float2> sun_azm_alt_h(kHoursPerYear);
    for (int k = 0; k < kHoursPerYear; ++k)
        sun_azm_alt_h[k] = make_float2(std::get<0>(bins.sun_azm_alt[k]), std::get<1>(bins.sun_azm_alt[k]));

    cudaSafeCall(cudaMemcpyToSymbol(sun_azm_alt, sun_azm_alt_h.data(), sizeof(float2) * kHoursPerYear));
    cudaSafeCall(cudaMemcpyToSymbol(sun_dense_idx_d, bins.sun_dense_idx.data(), sizeof(int) * kHoursPerYear));
    cudaSafeCall(cudaMemcpyToSymbol(direct_ir, direct_irradiance.data(), sizeof(float) * kHoursPerYear));
    cudaSafeCall(cudaMemcpyToSymbol(diffuse_ir, diffuse_irradiance.data(), sizeof(float) * kHoursPerYear));

    float hF_matrix[6][8] = { // Perez F matrix
		{ -0.0083117f ,	0.1299457f , 0.3296958f , 0.5682053f , 0.8730280f , 1.1326077f , 1.0601591f , 0.6777470f }, // f11
		{ 0.5877285f ,	0.6825954f,	0.4868735f , 0.1874525f , -0.3920403f ,	-1.2367284f , -1.5999137f ,	-0.3272588f }, // f12
		{ -0.0620636f ,	-0.1513752f , -0.2210958f ,	-0.2951290f , -0.3616149f, - 0.4118494f , -0.3589221f , -0.2504286f }, // f13
		{ -0.0596012f ,	-0.0189325f , 0.0554140f ,	0.1088631f , 0.2255647f , 0.2877813f ,	0.2642124f , 0.1561313f }, // f121
		{ 0.0721249f 	,	0.0659650f , -0.0639588f ,	-0.1519229f , -0.4620442f ,	-0.8230357f , -1.1272340f ,	-1.3765031f }, // f22
		{ -0.0220216f ,	-0.0288748f , -0.0260542f ,	-0.0139754f , 0.0012448f , 0.0558651f , 0.1310694f , 0.2506212f } // f23
	};
	cudaSafeCall(cudaMemcpyToSymbol(F_matrix, hF_matrix, sizeof(float) * 6 * 8, 0, cudaMemcpyHostToDevice));

}

} // namespace solar_gpu
