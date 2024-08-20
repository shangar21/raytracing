#include "headers/camera.cuh"
#include "headers/hittable.cuh"
#include "headers/material.cuh"
#include "headers/ray.cuh"
#include "headers/sphere.cuh"
#include "headers/vec3.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <time.h>

#define N_SAMPLES 10

const double inf = INFINITY;

__device__ point ray_color(Ray<double> &r, Hittable **world,
                           curandState *local_rand_state) {
  Ray<double> tmp_ray = r;
  point attenuation = point(1.0, 1.0, 1.0);
  for (int i = 0; i < 50; i++) {
    HitRecord rec;

    if ((*world)->hit(tmp_ray, Interval(0.001, inf), rec)) {
      Ray<double> scattered;
      point tmp_attenuation;
      if (rec.mat->scatter(tmp_ray, rec, attenuation, scattered,
                           local_rand_state)) {
        attenuation = attenuation * tmp_attenuation;
        tmp_ray = scattered;
      } else {
        return point(0.0, 0.0, 0.0);
      }
    } else {
      point unit_dir = unit_vector(tmp_ray.direction());
      double a = 0.5 * (unit_dir.y() + 1.0);
      point c = (1.0 - a) * point(1.0, 1.0, 1.0) + a * point(0.1, 0.1, 1.0);
      return attenuation * c;
    }
  }

  return point(0.0, 0.0, 0.0);
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int pixel_index = j * max_x + i;
  // Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render_kernel(double *R, double *G, double *B, int image_width,
                              int image_height, Camera **cam, Hittable **world,
                              curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= image_width || j >= image_height)
    return;

  int idx = j * image_width + i;
  curandState local_rand_state = rand_state[idx];

  for (int _ = 0; _ < N_SAMPLES; _++) {
    double x =
        double(i + curand_uniform(&local_rand_state)) / double(image_width);
    double y =
        double(j + curand_uniform(&local_rand_state)) / double(image_height);
    point ray_d = (*cam)->pixel00 + ((i + x) * (*cam)->pixel_delta_u) +
                  ((j + y) * (*cam)->pixel_delta_v);
    point ray_o = (*cam)->camera_center;
    Ray<double> r = Ray<double>(ray_o, ray_d);
    point color = ray_color(r, world, &local_rand_state);
    R[idx] += color.x();
    G[idx] += color.y();
    B[idx] += color.z();
  }

  R[idx] *= (*cam)->pixel_samples_scale;
  G[idx] *= (*cam)->pixel_samples_scale;
  B[idx] *= (*cam)->pixel_samples_scale;
}

__global__ void create_world(Hittable **d_list, Hittable **d_world,
                             Camera **d_camera) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_list[0] =
        new Sphere(point(0, 0, -1), 0.5, new Lambertian(point(0.8, 0.3, 0.3)));
    d_list[1] = new Sphere(point(0, -100.5, -1), 100,
                           new Lambertian(point(0.8, 0.8, 0.0)));
    d_list[2] =
        new Sphere(point(1, 0, -1), 0.5, new Metal(point(0.8, 0.6, 0.2), 1.0));
    d_list[3] =
        new Sphere(point(-1, 0, -1), 0.5, new Metal(point(0.8, 0.8, 0.8), 0.3));
    *d_world = new HittableList(d_list, 4);
    *d_camera = new Camera();
  }
}

__global__ void free_world(Hittable **d_list, Hittable **d_world,
                           Camera **d_camera) {
  for (int i = 0; i < 4; i++) {
    delete ((Sphere *)d_list[i])->mat;
    delete d_list[i];
  }
  delete *d_world;
  delete *d_camera;
}

// Function to initialize CUDA-related data and call the kernel
extern "C" void render_image(double *h_R, double *h_G, double *h_B,
                             int image_width, int image_height) {

  const int num_pixels = image_width * image_height;

  double *d_R, *d_G, *d_B;
  curandState *d_state;
  Hittable **d_list;
  Hittable **d_world;
  Camera **d_camera;

  // Allocate memory on the device
  cudaMalloc(&d_R, num_pixels * sizeof(double));
  cudaMalloc(&d_G, num_pixels * sizeof(double));
  cudaMalloc(&d_B, num_pixels * sizeof(double));
  cudaMalloc(&d_state, num_pixels * sizeof(curandState));
  cudaMalloc((void **)&d_list, 4 * sizeof(Hittable *));
  cudaMalloc((void **)&d_world, sizeof(Hittable *));
  cudaMalloc((void **)&d_camera, sizeof(Camera *));

  // Initialize world
  create_world<<<1, 1>>>(d_list, d_world, d_camera);

  // Initialize the random states
  render_init<<<dim3((image_width + 15) / 16, (image_height + 15) / 16),
                dim3(16, 16)>>>(image_width, image_height, d_state);

  // Launch the kernel to render the image
  render_kernel<<<dim3((image_width + 15) / 16, (image_height + 15) / 16),
                  dim3(16, 16)>>>(d_R, d_G, d_B, image_width, image_height,
                                  d_camera, d_world, d_state);

  // Copy the result back to the host
  cudaMemcpy(h_R, d_R, num_pixels * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_G, d_G, num_pixels * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, num_pixels * sizeof(double), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_R);
  cudaFree(d_G);
  cudaFree(d_B);
  cudaFree(d_state);
  free_world<<<1, 1>>>(d_list, d_world, d_camera);
}
