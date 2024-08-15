#ifndef HITTABLE_H
#define HITTABLE_H

#include "interval.h"
#include "ray.h"
#include "vec3.h"
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
  virtual bool hit(const Ray<double> &r, Interval<double> ray_t,
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

  bool hit(const Ray<double> &r, Interval<double> ray_t,
           HitRecord &rec) const override {
    HitRecord temp_rec;
    bool hit_anything = false;
    double closest_so_far = ray_t.max;

    for (hittable_ptr object : objects) {
      if (object->hit(r, Interval<double>(ray_t.min, closest_so_far),
                      temp_rec)) {
        hit_anything = true;
        closest_so_far = temp_rec.t;
        rec = temp_rec;
      }
    }

    return hit_anything;
  }
};

#endif
