#ifndef CAMERA_H
#define CAMERA_H

#include "ray.cuh"

class Camera {
public:
  double aspect_ratio = 16.0 / 9.0;
  int image_width = 1920;
  int samples_per_pixel = 3;
  int image_height = int(image_width / aspect_ratio);
  double fl = 1.0;
  double v_h = 2.0;
  double v_w = v_h * ((double)image_width / image_height);
  point camera_center = point(0.0, 0.0, 0.0);
  point pixel_delta_u;
  point pixel_delta_v;
  point pixel00;
  double pixel_samples_scale = 1.0 / samples_per_pixel;

  __device__ Camera() {
    point viewport_u = point(v_w, 0.0, 0.0);
    point viewport_v = point(0.0, -v_h, 0.0);

    pixel_delta_u = viewport_u / (double)image_width;
    pixel_delta_v = viewport_v / (double)image_height;

    point viewport_ul = camera_center - point(0.0, 0.0, fl) - viewport_u / 2.0 -
                        viewport_v / 2.0;

    pixel00 = viewport_ul + 0.5 * (pixel_delta_u + pixel_delta_v);
  }
};

#endif
