#ifndef RAY_H
#define RAY_H

#include "vec3.h"

template <typename T> class Ray {
private:
  vec3<T> orig;
  vec3<T> dir;

public:
  Ray() {}
  Ray(vec3<T> origin, vec3<T> direction) : orig(origin), dir(direction) {}

  Ray(const Ray &other) = default;
  Ray &operator=(const Ray &other) = default;

  Ray(Ray &&other) noexcept = default;
  Ray &operator=(Ray &&other) noexcept = default;

  const vec3<T> origin() const { return orig; }
  const vec3<T> direction() const { return dir; }

  vec3<T> at(T t) const { return orig + (t * dir); }
};

#endif
