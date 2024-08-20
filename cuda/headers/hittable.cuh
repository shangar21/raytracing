#ifndef HITTABLE_H
#define HITTABLE_H

#include "interval.cuh"
#include "ray.cuh"
#include "vec3.cuh"
#include <memory>
#include <vector>

class Material;

struct HitRecord {
  point p;
  point normal;
  double t;
  Material *mat;
};

class Hittable {
public:
  __device__ virtual bool hit(const Ray<double> &r, Interval ray_t,
                              HitRecord &rec) const = 0;
};

typedef Hittable **hittable_ptr;
typedef Hittable **hittable_list;

class HittableList : public Hittable {
public:
  hittable_list objects;
  int list_size = 0;

  __device__ HittableList() {}
  __device__ HittableList(hittable_ptr object, int n) {
    objects = object;
    list_size = n;
  }

  __device__ bool hit(const Ray<double> &r, Interval ray_t,
                      HitRecord &rec) const override {
    HitRecord temp_rec;
    bool hit_anything = false;
    double closest_so_far = ray_t.max;

    for (int i = 0; i < list_size; i++) {
      if (objects[i]->hit(r, Interval(ray_t.min, closest_so_far), temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }

    return hit_anything;
  }
};

#endif
