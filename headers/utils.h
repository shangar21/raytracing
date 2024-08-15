#ifndef UTILS_H
#define UTILS_H

#include "constants.h"
#include "Eigen/Dense"

inline double degrees_to_radians(double degrees) {
  return degrees * pi / 180.0;
}

template <typename T>
inline double numeric_infinity(){
	return std::numeric_limits<T>::infinity();
}

#endif
