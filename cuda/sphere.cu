#include "ray.cu"
#include "utils.cu"
#include <cuda_runtime.h>
#include <math.h>

struct sphere {
  double3 center;
  double radius;

  __host__ __device__ sphere(double3 c, double r) : center(c), radius(r) {}

  __host__ __device__ bool hit(const ray r, double &t, double3 &normal) {
    double3 oc = center - r.orig;
    double a = length_squared(r.dir);
    double b = -2.0 * dot(r.dir, oc);
    double c = length_squared(oc) - (radius * radius);
    double discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
      return false;
    }

    t = -b - sqrt(discriminant) / (2.0 * a);
    double3 p = r.at(t);
    double3 outward_normal = (p - center) / radius;
    bool front = dot(r.dir, outward_normal);
    normal = front ? outward_normal : -outward_normal;

    return true;
  }
};
