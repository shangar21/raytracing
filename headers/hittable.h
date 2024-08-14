#ifndef HITTABLE_H
#define HITTABLE_H

#include "vec3.h"
#include "ray.h"
#include <memory>
#include <vector>

class HitRecord {
public:
  point p;
  point normal;
  double t;
  bool front;

	HitRecord() {}

  void set_face_normal(const Ray<double> &r, const point &outward_normal) {
    front = dot(r.direction(), outward_normal);
    normal = front ? outward_normal : -outward_normal;
  }
};

class Hittable {
public:
  virtual bool hit(const Ray<double> &r, double ray_tmin, double ray_tmax,
                   HitRecord &rec) const = 0;
};

typedef std::shared_ptr<Hittable> hittable_ptr;
typedef std::vector<hittable_ptr> hittable_list;

class HittableList : public Hittable {
public:
  hittable_list objects;

  HittableList() {}
  HittableList(hittable_ptr object) { add(object); }

  void clear() { objects.clear(); }

  void add(hittable_ptr object) { objects.push_back(object); }

  bool hit(const Ray<double> &r, double ray_tmin, double ray_tmax,
           HitRecord &rec) const override{
    HitRecord temp_rec;
    bool hit_anything = false;
    double closest_so_far = ray_tmax;

    for (hittable_ptr object : objects) {
      if (object->hit(r, ray_tmin, closest_so_far, temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }

    return hit_anything;
  }
};

#endif
