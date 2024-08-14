#include "color.h"
#include "vec3.h"
#include <iostream>

int main() {
  int image_width = 1920;
  int image_height = 1080;

  std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

  for (int j = 0; j < image_height; j++) {
    std::clog << "\rScanlines remaining: " << (image_height - j) << ' '
              << std::flush;
    for (int i = 0; i < image_width; i++) {
      float r = double(i) / (image_width - 1);
      float g = double(j) / (image_height - 1);
      float b = 0.0;

      vec3<double> pixel = color(r, g, b);
      writeColor(std::cout, pixel);
    }
  }
  std::clog << "\rDone :)" << std::flush;
}
