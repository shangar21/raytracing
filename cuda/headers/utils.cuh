#ifndef UTILS_H
#define UTILS_H

#include "Eigen/Dense"
#include "constants.cuh"
#include <omp.h>

__host__ __device__ inline double degrees_to_radians(double degrees) {
  return degrees * pi / 180.0;
}

template <typename T> __host__ __device__ inline double numeric_infinity() {
  return std::numeric_limits<T>::infinity();
}

__host__ __device__ inline double random_double() {
  return std::rand() / (RAND_MAX + 1.0);
}

__host__ __device__ inline double random_double(double min, double max) {
  return min + (max - min) * random_double();
}

#endif
