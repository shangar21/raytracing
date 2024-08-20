#ifndef RAY_H
#define RAY_H

#include "vec3.cuh"

template <typename T> class Ray {
private:
  vec3<T> orig;
  vec3<T> dir;

public:
  __device__ Ray() {}
  __device__ Ray(vec3<T> origin, vec3<T> direction)
      : orig(origin), dir(direction) {}

  __device__ Ray(const Ray &other) = default;
  __device__ Ray &operator=(const Ray &other) = default;

  __device__ Ray(Ray &&other) noexcept = default;
  __device__ Ray &operator=(Ray &&other) noexcept = default;

  __device__ vec3<T> origin() const { return orig; }
  __device__ vec3<T> direction() const { return dir; }

  __device__ vec3<T> at(T t) const { return orig + (t * dir); }
};

#endif
