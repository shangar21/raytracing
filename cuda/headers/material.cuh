#ifndef MATERIAL_H
#define MATERIAL_H

#include <curand_kernel.h>
#include "hittable.cuh"
#include "ray.cuh"
#include "vec3.cuh"

#define RANDVEC3 point(curand_uniform(local_rand_state), curand_uniform(local_rand_state),\
        curand_uniform(local_rand_state))

__device__ point random_in_unit_sphere(curandState *local_rand_state) {
  point p;
  do {
    p = RANDVEC3 * 2.0 - point(1.0, 1.0, 1.0);

  } while (p.length_squared() >= 1.0);
  return p;
}

class Material {
public:
  __device__ virtual ~Material() = default;

  __device__ virtual bool scatter(const Ray<double> &r_in, const HitRecord &rec,
                                  point &attenuation, Ray<double> &scattered,
                                  curandState *local_rand_state) const {
    return false;
  }
};

class Lambertian : public Material {
public:
  __device__ Lambertian(const point &albedo) : albedo(albedo) {}

  __device__ bool scatter(const Ray<double> &r_in, const HitRecord &rec,
                          point &attenuation, Ray<double> &scattered,
                          curandState *local_rand_state) const override {
    point scatter_dir = rec.normal + random_in_unit_sphere(local_rand_state);
    scattered = Ray<double>(rec.p, scatter_dir);
    attenuation = albedo;
    return true;
  }

private:
  point albedo;
};

class Metal : public Material {
public:
  double fuzz = 0.0;
  __device__ Metal(const point &albedo) : albedo(albedo) {}
  __device__ Metal(const point &albedo, double f) : albedo(albedo) {
    if (f < 1)
      fuzz = f;
  }

  __device__ bool scatter(const Ray<double> &r_in, const HitRecord &rec,
                          point &attenuation, Ray<double> &scattered,
                          curandState *local_rand_state) const override {
    point reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered =
        Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
    attenuation = albedo;
    return true;
  }

private:
  point albedo;
};

#endif
