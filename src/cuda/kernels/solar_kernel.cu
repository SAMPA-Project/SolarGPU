#include "cuda/kernels/solar_kernel.cuh"
#include "cuda/device_globals.cuh"
#include "pipeline_math_constants.h"

using solar_gpu::grid_s_type;
using solar_gpu::get_svf_bit;

__device__ inline float Perez(const float& cos_z_angle, const float& cos_angle_of_incidence, const float& dif_irad, const float& dir_irad, const float& slope, const int& n) {
	// relative optical air mass
	float z_deg = acosf(cos_z_angle) * (180.0f / (float)M_PI);
	float AM = 1.0f / (cos_z_angle + 0.50572f * powf(96.07995f - z_deg, -1.6364f)); // Kasten-Young (1989)

    float a = max(0.0f, (cos_angle_of_incidence));
	float b = max(0.08715574f, cos_z_angle);

	float a_div_b = 0;
	if (b > 0) a_div_b = a / b;

	// zenith angle
	float z_angle = acosf(cos_z_angle);

	// Perez sky clearness
	float eps = ((dif_irad + dir_irad / cosf(z_angle)) / dif_irad + 1.041f * (z_angle * z_angle * z_angle)) / (1.0f + 1.041f * z_angle * z_angle * z_angle);

	// Perez sky brightnesss
	float B = 2 * (float)M_PI*(n / 365.0f);
	float E0 = 1367 * (1.00011f + 0.034221f*cos(B) + 0.00128f*sin(B) + 0.000719f*cos(2 * B) + 0.000077f*sin(2 * B)); // irradiance on top of atmosphere (not solar constant)

	float delta = (AM * dif_irad) / E0;

	eps = max(1.0f, eps); 

	int eps_ind = 7;
	if (eps < 1.065f) eps_ind = 0;
	else if (eps < 1.230f) eps_ind = 1;
	else if (/*eps >= 1.230f &&*/ eps < 1.500f) eps_ind = 2;
	else if (/*eps >= 1.500f && */ eps < 1.950f) eps_ind = 3;
	else if (/*eps >= 1.950f && */ eps < 2.800f) eps_ind = 4;
	else if (/*eps >= 2.800f && */ eps < 4.500f) eps_ind = 5;
	else if (/*eps >= 4.500f && */ eps < 6.200f) eps_ind = 6;

	// Perez F1 = circumsolar brightness, F2 = horizon brightness
	float F1 = max(0.0f, F_matrix[0][eps_ind] + F_matrix[1][eps_ind] * delta + F_matrix[2][eps_ind] * z_angle);
	float F2 = F_matrix[3][eps_ind] + F_matrix[4][eps_ind] * delta + F_matrix[5][eps_ind] * z_angle;

	// Modified anisotropic Perez irradiance model
    float dif = dif_irad * ((1 - F1) * ((1 + cos(slope)) / 2)  + F1 * a_div_b + F2 *  sin(slope));

	return(dif);
}

/* Uses the precomputed dense SVF bin index (sun_dense_idx_d) for the hour
   being evaluated, rather than recomputing the (i,j) hemisphere bin here. */
__global__ void calc_kernel_solar(grid_s_type* grid_s, float* hourly_data,
    float* slopes, float* aspects,
    int n, int h, int total_pixels)
{
    const int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_id >= total_pixels) return;

    int ind = n * 24 + h;
    //if (sun_azm_alt[ind].y < 1.0f) return;

    int di = sun_dense_idx_d[ind];

    float slope = slopes[t_id], aspect = aspects[t_id];
    float cos_z_angle = cosf((90.0f - sun_azm_alt[ind].y) * solar_gpu::kDegToRad);
    float cos_aoi =  cos_z_angle * cosf(slope) + sinf(acosf(cos_z_angle)) * sinf(slope) * cosf(sun_azm_alt[ind].x * solar_gpu::kDegToRad - aspect);
    float I = 0.0f;

    if (cos_aoi > 0.0f && di >= 0 && !get_svf_bit((size_t)t_id, (size_t)di, grid_s, num_data))
        I += fmaxf(0.0f, fmaxf(direct_ir[ind], 1e-3f) * (cos_aoi));
  
    I += fmaxf(0.0f, Perez(cos_z_angle, cos_aoi, diffuse_ir[ind], direct_ir[ind], slope, n));
        
    hourly_data[t_id] = I;
}

__global__ void accumulate_kernel(float* __restrict__ total,
    const float* __restrict__ hourly, const float* __restrict__ reflected, int total_pixels)
{
    const int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_id >= total_pixels) return;
    total[t_id] += reflected[t_id] + hourly[t_id];
}
