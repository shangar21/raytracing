#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#define N_SAMPLES 3

__host__ __device__ bool hit(double radius, double *center, double *ray_o,
                             double *ray_d) {
  double oc_x = center[0] - ray_o[0];
  double oc_y = center[1] - ray_o[1];
  double oc_z = center[2] - ray_o[2];

  double a = ray_d[0] * ray_d[0] + ray_d[1] * ray_d[1] + ray_d[2] * ray_d[2];
  double b = -2.0 * (ray_d[0] * oc_x + ray_d[1] * oc_y + ray_d[2] * oc_z);
  double c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - (radius * radius);

  double discriminant = b * b - 4 * a * c;

  return discriminant >= 0;
}

__global__ void init_curand_state(curandState *state, unsigned long long seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, idx, 0, &state[idx]);
}

__global__ void render_kernel(
    double *R, double *G, double *B, curandState *state, int image_width,
    int image_height, double pixel00_x, double pixel00_y, double pixel00_z,
    double pixel_delta_u_x, double pixel_delta_u_y, double pixel_delta_u_z,
    double pixel_delta_v_x, double pixel_delta_v_y, double pixel_delta_v_z,
    double radius, double sphere_o_x, double sphere_o_y, double sphere_o_z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= image_width || j >= image_height)
    return;

  int idx = j * image_width + i;

  double r, g, b;
  double ray_o_x, ray_o_y, ray_o_z;
  double ray_d_x, ray_d_y, ray_d_z;
  double center[3] = {sphere_o_x, sphere_o_y, sphere_o_z};

  curandState localState = state[idx];

  for (int _ = 0; _ < N_SAMPLES; _++) {
    double offset_x = curand_uniform_double(&localState) - 0.5;
    double offset_y = curand_uniform_double(&localState) - 0.5;
    double offset_z = curand_uniform_double(&localState) - 0.5;
    ray_o_x = 0.0;
    ray_o_y = 0.0;
    ray_o_z = 0.0;
    ray_d_x = pixel00_x + ((i + offset_x) * pixel_delta_u_x) +
              ((j + offset_y) * pixel_delta_v_x);
    ray_d_y = pixel00_y + ((i + offset_y) * pixel_delta_u_y) +
              ((j + offset_y) * pixel_delta_v_y);
    ray_d_z = pixel00_z + ((i + offset_z) * pixel_delta_u_z) +
              ((j + offset_y) * pixel_delta_v_z);

    double ray_o[3] = {ray_o_x, ray_o_y, ray_o_z};
    double ray_d[3] = {ray_d_x, ray_d_y, ray_d_z};

    // TO-DO: Add code to check if ray defined by ray_o_* and ray_d_* intersects
    // a sphere
    if (hit(radius, center, ray_o, ray_d)) {
      R[idx] = 1.0;
      G[idx] = 0.0;
      B[idx] = 0.5;
    }
  }
}

// Function to initialize CUDA-related data and call the kernel
extern "C" void
render_image(double *h_R, double *h_G, double *h_B, int image_width,
             int image_height, double pixel00_x, double pixel00_y,
             double pixel00_z, double pixel_delta_u_x, double pixel_delta_u_y,
             double pixel_delta_u_z, double pixel_delta_v_x,
             double pixel_delta_v_y, double pixel_delta_v_z, double radius,
             double sphere_o_x, double sphere_o_y, double sphere_o_z) {

  const int num_pixels = image_width * image_height;

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
                  dim3(16, 16)>>>(
      d_R, d_G, d_B, d_state, image_width, image_height, pixel00_x, pixel00_y,
      pixel00_z, pixel_delta_u_x, pixel_delta_u_y, pixel_delta_u_z,
      pixel_delta_v_x, pixel_delta_v_y, pixel_delta_v_z, radius, sphere_o_x,
      sphere_o_y, sphere_o_z);

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
