#ifndef INTERVAL_H
#define INTERVAL_H

#include "utils.h"

template <typename T> class Interval {
public:
  double min, max;

  Interval() : min(numeric_infinity<T>()), max(-numeric_infinity<T>()) {}

  Interval(T min, T max) : min(min), max(max) {}

  double size() const { return max - min; }

  bool contains(T x) { return min <= x && x <= max; }

  bool surrounds(T x) { return min < x && x < max; }

  Interval get_universe() {
    return Interval(-numeric_infinity<T>(), numeric_infinity<T>());
  }

  Interval get_empty() { return Interval(); }

  T clamp(T x) const {
    if (x < min)
      return min;
    if (x > max)
      return max;
    return x;
  }
};

#endif
