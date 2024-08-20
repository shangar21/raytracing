#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.cuh"
#include "ray.cuh"

class Sphere : public Hittable {
public:
  double radius;
  point center;
  Material *mat;

  __device__ Sphere(point center, double radius, Material *mat)
      : center(center), radius(radius), mat(mat) {}

  __device__ bool hit(const Ray<double> &r, Interval ray_t,
                      HitRecord &rec) const {
    point oc = center - r.origin();
    double a = r.direction().length_squared();
    double b = -2.0 * dot(r.direction(), oc);
    double c = oc.length_squared() - (radius * radius);
    double discriminant = b * b - 4 * a * c;

    if (discriminant >= 0) {
      double t1 = (-b - std::sqrt(discriminant)) / (2.0 * a);
      double t2 = (-b + std::sqrt(discriminant)) / (2.0 * a);

      if (ray_t.surrounds(t1) || ray_t.surrounds(t2)) {
        double t = (ray_t.surrounds(t1)) ? t1 : t2;
        point p = r.at(t);
        point normal = (p - center) / radius;
        rec.t = t;
        rec.p = p;
        rec.normal = dot(r.direction(), normal) < 0 ? normal : -normal;
        rec.mat = mat;
        return true;
      }
    }

    return false;
  }
};

#endif
