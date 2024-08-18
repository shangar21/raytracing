#include "raytracing.h"

int main() {
  HittableList world;

  world.add(std::make_shared<Sphere>(point(0, -100.5, -1), 100));
  world.add(std::make_shared<Sphere>(point(0, 0, -1), 0.5));

  Camera cam;

  cam.render_png(world);
}
