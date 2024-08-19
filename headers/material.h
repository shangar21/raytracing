#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "ray.h"
#include "vec3.h"

class Material {
public:
  virtual ~Material() = default;

  virtual bool scatter(const Ray<double> &r_in, const HitRecord &rec,
                       point &attenuation, Ray<double> &scattered) const {
    return false;
  }
};

class Lambertian : public Material {
public:
  Lambertian(const point &albedo) : albedo(albedo) {}

  bool scatter(const Ray<double> &r_in, const HitRecord &rec,
               point &attenuation, Ray<double> &scattered) const override {
    point scatter_dir = rec.normal + random_unit_vector<double>();

    if (scatter_dir.near_zero())
      scatter_dir = rec.normal;

    scattered = Ray<double>(rec.p, scatter_dir);
    attenuation = albedo;
    return true;
  }

private:
  point albedo;
};

class Metal : public Material {
public:
  Metal(const point &albedo) : albedo(albedo) {}

  bool scatter(const Ray<double> &r_in, const HitRecord &rec,
               point &attenuation, Ray<double> &scattered) const override {
    point reflected = reflect(r_in.direction(), rec.normal);
    scattered = Ray(rec.p, reflected);
    attenuation = albedo;
    return true;
  }

private:
  point albedo;
};

#endif
