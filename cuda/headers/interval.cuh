#ifndef INTERVAL_H
#define INTERVAL_H

#include "utils.cuh"

class Interval {
public:
  double min, max;

  __host__ __device__ Interval()
      : min(numeric_infinity<double>()), max(-numeric_infinity<double>()) {}

  __host__ __device__ Interval(double min, double max) : min(min), max(max) {}

  __host__ __device__ double size() const { return max - min; }

  __host__ __device__ bool contains(double x) { return min <= x && x <= max; }

  __host__ __device__ bool surrounds(double x) { return min < x && x < max; }

  __host__ __device__ Interval get_universe() {
    return Interval(-numeric_infinity<double>(), numeric_infinity<double>());
  }

  __host__ __device__ Interval get_empty() { return Interval(); }

  __host__ __device__ double clamp(double x) const {
    if (x < min)
      return min;
    if (x > max)
      return max;
    return x;
  }
};

#endif
