#include "ray.h"
#include <iostream>

int main() {
  // Create vectors for the origin and direction
  vec3<double> origin(0.0, 0.0, 0.0);
  vec3<double> direction(1.0, 2.0, 3.0);

  // Create a ray with the above origin and direction
  Ray<double> r(origin, direction);

  // Test origin
  vec3<double> test_origin = r.origin();
  std::cout << "Ray origin: " << test_origin << std::endl;

  // Test direction
  vec3<double> test_direction = r.direction();
  std::cout << "Ray direction: " << test_direction << std::endl;

  // Test the `at` method for different values of t
  double t1 = 0.0;
  vec3<double> point1 = r.at(t1);
  std::cout << "Point at t = " << t1 << ": " << point1 << std::endl;

  double t2 = 1.0;
  vec3<double> point2 = r.at(t2);
  std::cout << "Point at t = " << t2 << ": " << point2 << std::endl;

  double t3 = 2.0;
  vec3<double> point3 = r.at(t3);
  std::cout << "Point at t = " << t3 << ": " << point3 << std::endl;

  // Test with a negative t
  double t4 = -1.0;
  vec3<double> point4 = r.at(t4);
  std::cout << "Point at t = " << t4 << ": " << point4 << std::endl;

  return 0;
}
