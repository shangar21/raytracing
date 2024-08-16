#include <cuda_runtime.h>
#include <math.h>

// Dot product of two double3 vectors
__host__ __device__ double dot(const double3 &a, const double3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross product of two double3 vectors
__host__ __device__ double3 cross(const double3 &a, const double3 &b) {
  return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x);
}

// Length (magnitude) of a double3 vector
__host__ __device__ double length(const double3 &v) { return sqrt(dot(v, v)); }

// Normalize a double3 vector (unit vector)
__host__ __device__ double3 unit_vector(const double3 &v) {
  double len = length(v);
  return make_double3(v.x / len, v.y / len, v.z / len);
}

// Convert double* to double3
__host__ __device__ double3 doublePtrToDouble3(const double *ptr) {
  return make_double3(ptr[0], ptr[1], ptr[2]);
}

// Length squared of a double3 vector
__host__ __device__ double length_squared(const double3 &v) {
  return dot(v, v);
}

__host__ __device__ double clamp(const double v, double l, double u){
	if (v < l) return l;
	if (v > u) return u;
	return v;
}

// Addition of two double3 vectors
__host__ __device__ double3 operator+(const double3 &a, const double3 &b) {
  return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// Subtraction of two double3 vectors
__host__ __device__ double3 operator-(const double3 &a, const double3 &b) {
  return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Multiplication of a double3 vector by a scalar
__host__ __device__ double3 operator*(const double3 &v, double scalar) {
  return make_double3(v.x * scalar, v.y * scalar, v.z * scalar);
}

// Multiplication of a scalar by a double3 vector
__host__ __device__ double3 operator*(double scalar, const double3 &v) {
  return make_double3(v.x * scalar, v.y * scalar, v.z * scalar);
}

// Division of a double3 vector by a scalar
__host__ __device__ double3 operator/(const double3 &v, double scalar) {
  return make_double3(v.x / scalar, v.y / scalar, v.z / scalar);
}

// Compound assignment: addition
__host__ __device__ double3 &operator+=(double3 &a, const double3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

// Compound assignment: subtraction
__host__ __device__ double3 &operator-=(double3 &a, const double3 &b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

// Compound assignment: multiplication by a scalar
__host__ __device__ double3 &operator*=(double3 &v, double scalar) {
  v.x *= scalar;
  v.y *= scalar;
  v.z *= scalar;
  return v;
}

// Compound assignment: division by a scalar
__host__ __device__ double3 &operator/=(double3 &v, double scalar) {
  v.x /= scalar;
  v.y /= scalar;
  v.z /= scalar;
  return v;
}

// Unary minus (negation) for double3
__host__ __device__ double3 operator-(const double3 &v) {
  return make_double3(-v.x, -v.y, -v.z);
}
