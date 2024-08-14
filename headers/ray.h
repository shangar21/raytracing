#ifndef RAY_H
#define RAY_H

#include "vec3.h"

template <typename T> class ray {
private:
  const vec3<T> orig;
  const vec3<T> dir;

public:
  ray(vec3<T> origin, vec3<T> direction) : orig(origin), dir(direction) {}

  const vec3<T> origin() const { return orig; }
  const vec3<T> direction() const { return dir; }

  vec3<T> at(T t) const { return orig + (t * dir); }
};

#endif
