#include <cuda_runtime.h>

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
