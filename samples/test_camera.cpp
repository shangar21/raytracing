#include "raytracing.h"

int main() {
  HittableList world;

  Lambertian mat_ground(color(0.1, 1.0, 0.2));
  Metal mat_obj(color(1.0, 0.2, 0.2));

  std::shared_ptr<Lambertian> mat_ground_ptr =
      std::make_shared<Lambertian>(mat_ground);
  std::shared_ptr<Metal> mat_obj_ptr =
      std::make_shared<Metal>(mat_obj);

  world.add(std::make_shared<Sphere>(point(0, 0, -1), 0.5, mat_obj_ptr));
  world.add(
      std::make_shared<Sphere>(point(0, -100.5, -1), 100, mat_ground_ptr));

  Camera cam;

  cam.render_png(world);
}
