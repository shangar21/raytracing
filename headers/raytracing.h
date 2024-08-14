#ifndef RAYTRACING_H
#define RAYTRACING_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

// Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions
inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

// Headers
#include "vec3.h"
#include "ray.h"
#include "color.h"
#include "hittable.h"

#endif
