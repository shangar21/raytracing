#ifndef VEC3_H
#define VEC3_H

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

  vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
  T operator[](int i) const { return e[i]; }

  vec3 &operator+=(const vec3 &v) {
    e[0] += v.x();
    e[1] += v.y();
    e[2] += v.z();
  }

  vec3 &operator*=(const vec3 &v) {
    e[0] *= v.x();
    e[1] *= v.y();
    e[2] *= v.z();
  }

  vec3 &operator/=(const vec3 &v) {
    e[0] /= v.x();
    e[1] /= v.y();
    e[2] /= v.z();
  }

  T length() const {
    return static_cast<T>(std::sqrt(static_cast<double>(length_squared())));
  }

  T length_squared() const {
    return (e[0] * e[0]) + (e[1] * e[1]) + (e[2] * e[2]);
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

typedef vec3<double> point;
typedef vec3<double> color;
#endif
