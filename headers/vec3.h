#ifndef VEC3_H
#define VEC3_H

#include "utils.h"
#include <cmath>
#include <iostream>

template <typename T> class vec3 {
public:
  T e[3];

  vec3() : e{0.0, 0.0, 0.0} {}
  vec3(T x, T y, T z) : e{x, y, z} {}

  T x() const { return e[0]; }
  T y() const { return e[1]; }
  T z() const { return e[2]; }

  vec3(const vec3 &other) = default;
  vec3 &operator=(const vec3 &other) = default;

  vec3(vec3 &&other) noexcept = default;
  vec3 &operator=(vec3 &&other) noexcept = default;

  vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
  T operator[](int i) const { return e[i]; }

  vec3 &operator+=(const vec3 &v) {
    e[0] += v.x();
    e[1] += v.y();
    e[2] += v.z();
    return *this;
  }

  vec3 &operator*=(const vec3 &v) {
    e[0] *= v.x();
    e[1] *= v.y();
    e[2] *= v.z();
    return *this;
  }

  vec3 &operator/=(const vec3 &v) {
    e[0] /= v.x();
    e[1] /= v.y();
    e[2] /= v.z();
    return *this;
  }

  T length() const {
    return static_cast<T>(std::sqrt(static_cast<double>(length_squared())));
  }

  T length_squared() const {
    return (e[0] * e[0]) + (e[1] * e[1]) + (e[2] * e[2]);
  }

  static vec3 random() {
    return vec3(random_double(), random_double(), random_double());
  }

  static vec3 random(T min, T max) {
    return vec3(random_double(min, max), random_double(min, max),
                random_double(min, max));
  }

  bool near_zero() const {
    double s = 1e-8;
    return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) &&
           (std::fabs(e[2]) < s);
  }
};

// Output stream operator
template <typename T>
inline std::ostream &operator<<(std::ostream &out, const vec3<T> &v) {
  return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

// Addition operator
template <typename T>
inline vec3<T> operator+(const vec3<T> &u, const vec3<T> &v) {
  return vec3<T>(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

// Subtraction operator
template <typename T>
inline vec3<T> operator-(const vec3<T> &u, const vec3<T> &v) {
  return vec3<T>(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

// Multiplication operator (element-wise)
template <typename T>
inline vec3<T> operator*(const vec3<T> &u, const vec3<T> &v) {
  return vec3<T>(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

// Multiplication by scalar
template <typename T> inline vec3<T> operator*(T t, const vec3<T> &v) {
  return vec3<T>(t * v.e[0], t * v.e[1], t * v.e[2]);
}

// Multiplication by scalar (reverse order)
template <typename T> inline vec3<T> operator*(const vec3<T> &v, T t) {
  return t * v;
}

// Division operator (element-wise)
template <typename T>
inline vec3<T> operator/(const vec3<T> &u, const vec3<T> &v) {
  return vec3<T>(u.e[0] / v.e[0], u.e[1] / v.e[1], u.e[2] / v.e[2]);
}

// Division by scalar
template <typename T> inline vec3<T> operator/(vec3<T> v, T t) {
  return (1 / t) * v;
}

// Dot product
template <typename T> inline T dot(const vec3<T> &u, const vec3<T> &v) {
  return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

// Cross product
template <typename T> inline vec3<T> cross(const vec3<T> &u, const vec3<T> &v) {
  return vec3<T>(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                 u.e[2] * v.e[0] - u.e[0] * v.e[2],
                 u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

template <typename T> inline vec3<T> unit_vector(const vec3<T> &v) {
  return v / v.length();
}

template <typename T> inline vec3<T> random_in_unit_sphere() {
  while (true) {
    vec3<T> p = vec3<T>::random(-1.0, 1.0);
    if (p.length_squared() < 1) {
      return p;
    }
  }
}

template <typename T> inline vec3<T> random_unit_vector() {
  return unit_vector(random_in_unit_sphere<T>());
}

template <typename T>
inline vec3<T> random_on_hemisphere(const vec3<T> &normal) {
  vec3<T> on_unit_sphere = random_unit_vector<T>();
  return dot(on_unit_sphere, normal) > 0.0 ? on_unit_sphere : -on_unit_sphere;
}

template <typename T>
inline vec3<T> reflect(const vec3<T> &v, const vec3<T> &n) {
  return v - 2 * dot(v, n) * n;
}

typedef vec3<double> point;
typedef vec3<double> color;
#endif
