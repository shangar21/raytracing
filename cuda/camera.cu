#include "ray.cu"
#include "sphere.cu"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#define N_SAMPLES 3

__global__ void init_curand_state(curandState *state, unsigned long long seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, idx, 0, &state[idx]);
}

__global__ void render_kernel(double *R, double *G, double *B,
                              curandState *state, int image_width,
                              int image_height, double3 pixel00,
                              double3 pixel_delta_u, double3 pixel_delta_v,
                              sphere s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= image_width || j >= image_height)
    return;

  int idx = j * image_width + i;

  double ray_d_x, ray_d_y, ray_d_z;

  curandState localState = state[idx];

  for (int _ = 0; _ < N_SAMPLES; _++) {
    double offset_x = curand_uniform_double(&localState) - 0.5;
    double offset_y = curand_uniform_double(&localState) - 0.5;
    double offset_z = curand_uniform_double(&localState) - 0.5;
    ray_d_x = pixel00.x + ((i + offset_x) * pixel_delta_u.x) +
              ((j + offset_x) * pixel_delta_v.x);
    ray_d_y = pixel00.y + ((i + offset_y) * pixel_delta_u.y) +
              ((j + offset_y) * pixel_delta_v.y);
    ray_d_z = pixel00.z + ((i + offset_z) * pixel_delta_u.z) +
              ((j + offset_z) * pixel_delta_v.z);

    double3 ray_o = make_double3(0.0, 0.0, 0.0);
    double3 ray_d = make_double3(ray_d_x, ray_d_y, ray_d_z);

    ray r = ray(ray_o, ray_d);

    double t = 0.0;
    double3 normal = make_double3(1.0, 0.0, 0.5);

    if (s.hit(r, t, normal)) {
      R[idx] = 0.5 * (normal.x + 1);
      G[idx] = 0.5 * (normal.y + 1);
      B[idx] = 0.5 * (normal.z + 1);
    } else {
			double3 unit_direction = unit_vector(r.dir);
			double a = 0.5 * (unit_direction.y + 1.0);
			R[idx] = (1.0 - a) + (a * 1.0);
			G[idx] = (1.0 - a) + (a * 0.1);
			B[idx] = (1.0 - a) + (a * 0.1);
		}
  }
}

// Function to initialize CUDA-related data and call the kernel
extern "C" void render_image(double *h_R, double *h_G, double *h_B,
                             int image_width, int image_height,
                             double *pixel00_ptr, double *pixel_delta_u_ptr,
                             double *pixel_delta_v_ptr, double radius,
                             double *center_ptr) {

  const int num_pixels = image_width * image_height;

  double3 pixel00 = doublePtrToDouble3(pixel00_ptr);
  double3 pixel_delta_u = doublePtrToDouble3(pixel_delta_u_ptr);
  double3 pixel_delta_v = doublePtrToDouble3(pixel_delta_v_ptr);
  double3 center = doublePtrToDouble3(center_ptr);

  sphere s = sphere(center, radius);

  double *d_R, *d_G, *d_B;
  curandState *d_state;

  // Allocate memory on the device
  cudaMalloc(&d_R, num_pixels * sizeof(double));
  cudaMalloc(&d_G, num_pixels * sizeof(double));
  cudaMalloc(&d_B, num_pixels * sizeof(double));
  cudaMalloc(&d_state, num_pixels * sizeof(curandState));

  // Initialize the random states
  init_curand_state<<<dim3((image_width + 15) / 16, (image_height + 15) / 16),
                      dim3(16, 16)>>>(d_state, time(NULL));

  // Launch the kernel to render the image
  render_kernel<<<dim3((image_width + 15) / 16, (image_height + 15) / 16),
                  dim3(16, 16)>>>(d_R, d_G, d_B, d_state, image_width,
                                  image_height, pixel00, pixel_delta_u,
                                  pixel_delta_v, s);

  // Copy the result back to the host
  cudaMemcpy(h_R, d_R, num_pixels * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_G, d_G, num_pixels * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, num_pixels * sizeof(double), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_R);
  cudaFree(d_G);
  cudaFree(d_B);
  cudaFree(d_state);
}
