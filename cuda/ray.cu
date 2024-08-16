#ifndef RAY_H
#define RAY_H

#include <cuda_runtime.h>

struct ray {
  double3 orig;
  double3 dir;

  __host__ __device__ ray(double3 origin, double3 direction)
      : orig(origin), dir(direction) {}

  __host__ __device__ double3 at(double t) const {
    double at_x = orig.x + t * dir.x;
    double at_y = orig.y + t * dir.y;
    double at_z = orig.z + t * dir.z;
    return make_double3(at_x, at_y, at_z);
  }
};

#endif
