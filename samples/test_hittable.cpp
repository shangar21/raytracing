#include "raytracing.h"

color ray_color(const Ray<double> &r, const Hittable &world) {
  HitRecord rec;

  if (world.hit(r, Interval<double>(0.0, infinity), rec)) {
    return 0.5 * (rec.normal + color(1.0, 1.0, 1.0));
  }

  point unit_direction = unit_vector(r.direction());
  double a = 0.5 * (unit_direction.y() + 1.0);
  return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(1.0, 0.1, 0.1);
}

int main() {
  double aspect_ratio = 16.0 / 9.0;
  int image_width = 1920;

  int image_height = int(image_width / aspect_ratio);
  image_height = (image_height < 1) ? 1 : image_height;

  HittableList world;
  world.add(std::make_shared<Sphere>(point(0.0, 0.0, -1.0), 0.5));
  world.add(std::make_shared<Sphere>(point(0.0, -100.5, -1.0), 100));

  double fl = 1.0;
  double v_h = 2.0;
  double v_w = v_h * ((double)image_width / image_height);
  point camera_center = point(0.0, 0.0, 0.0);

  point viewport_u = point(v_w, 0.0, 0.0);
  point viewport_v = point(0.0, -v_h, 0.0);

  point pixel_delta_u = viewport_u / (double)image_width;
  point pixel_delta_v = viewport_v / (double)image_height;

  point viewport_ul =
      camera_center - point(0, 0, fl) - viewport_u / 2.0 - viewport_v / 2.0;
  point pixel_00 = viewport_ul + 0.5 * (pixel_delta_u + pixel_delta_v);

  std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

  for (int j = 0; j < image_height; j++) {
    for (int i = 0; i < image_width; i++) {
      point pixel_center =
          pixel_00 + ((double)i * pixel_delta_u) + ((double)j * pixel_delta_v);
      point ray_dir = pixel_center - camera_center;
      Ray r = Ray(camera_center, ray_dir);
      point pixel = ray_color(r, world);
      writeColor(std::cout, pixel);
    }
  }
}
