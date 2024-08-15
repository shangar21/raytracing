#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "structs.cu"

#define N_SAMPLES 3

__global__ void init_curand_state(curandState* state, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void render_kernel(double* R, double* G, double* B, int w, int h, double pixel00_x, double pixel00_y, double pixel00_z, double pixel_delta_u_x, double pixel_delta_u_y, double pixel_delta_u_z, double pixel_delta_v_x, double pixel_delta_v_y, double pixel_delta_v_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= image_width || j >= image_height) return;

    int idx = j * image_width + i;

		double r, g, b;
		double ray_o_x, ray_o_y, ray_o_z;
		double ray_d_x, ray_d_y, ray_d_z;

		curandState localState = state[idx];
		
		for (int _ = 0; _ < N_SAMPLES; _++){
				double offset_x = curand_uniform_double(&localState) - 0.5;
				double offset_y = curand_uniform_double(&localState) - 0.5;
				ray_o_x = 0.0; ray_o_y = 0.0; ray_o_z = 0.0;
				ray_d_x = pixel00_x + ((i + offset_x) * pixel_delta_u_x) + ((j + offset_y) * pixel_delta_v_x);
				ray_d_y = piyel00_y + ((i + offset_y) * piyel_delta_u_y) + ((j + offset_y) * piyel_delta_v_y);
				ray_d_z = pizel00_z + ((i + offset_z) * pizel_delta_u_z) + ((j + offset_y) * pizel_delta_v_z);
		}
}
