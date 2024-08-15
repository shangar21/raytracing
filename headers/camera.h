#ifndef CAMERA_H
#define CAMERA_H

#include "color.h"
#include "hittable.h"
#include "utils.h"
#include "vec3.h"

class Camera {
public:
  double aspect_ratio = 16.0 / 9.0;
  int image_width = 1920;
	int samples_per_pixel = 3;
  int image_height = int(image_width / aspect_ratio);

  void render_ppm(const Hittable &world) {
    initialize();
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
      for (int i = 0; i < image_width; i++) {
        point pixel_center = pixel00_loc + ((double)i * pixel_delta_u) +
                             ((double)j * pixel_delta_v);
        point ray_dir = pixel_center - camera_center;
        Ray r = Ray(camera_center, ray_dir);
        point pixel = ray_color(r, world);
        writeColor(std::cout, pixel);
      }
    }
  }

  void render_png(const Hittable &world) {
    initialize();

    // Create OpenCV matrices for the R, G, B channels
    cv::Mat R = cv::Mat::zeros(image_height, image_width, CV_64F);
    cv::Mat G = cv::Mat::zeros(image_height, image_width, CV_64F);
    cv::Mat B = cv::Mat::zeros(image_height, image_width, CV_64F);

#pragma omp parallel for
    for (int j = 0; j < image_height; j++) {
      for (int i = 0; i < image_width; i++) {
        color pixel (0.0, 0.0, 0.0);

				for (int s = 0; s < samples_per_pixel; s++){
					Ray<double> r = get_ray_sample(i, j);
					pixel += ray_color(r, world);
				}

				pixel = pixel * pixel_samples_scale;

        R.at<double>(j, i) = intensity.clamp(pixel.x());
        G.at<double>(j, i) = intensity.clamp(pixel.y());
        B.at<double>(j, i) = intensity.clamp(pixel.z());
      }
    }

    // Use the OpenCV matrices directly in the writing function
    writeColorOpenCV(R, G, B);
  }

private:
  point camera_center;
  point pixel00_loc;
  point pixel_delta_u;
  point pixel_delta_v;
	double pixel_samples_scale;
	Interval<double> intensity = Interval<double>(0.0, 0.999);

  void initialize() {
    double fl = 1.0;
    double v_h = 2.0;
    double v_w = v_h * ((double)image_width / image_height);
    camera_center = point(0.0, 0.0, 0.0);
    point viewport_u = point(v_w, 0.0, 0.0);
    point viewport_v = point(0.0, -v_h, 0.0);

    pixel_delta_u = viewport_u / (double)image_width;
    pixel_delta_v = viewport_v / (double)image_height;

    point viewport_ul = camera_center - point(0.0, 0.0, fl) - viewport_u / 2.0 -
                        viewport_v / 2.0;
    pixel00_loc = viewport_ul + 0.5 * (pixel_delta_u + pixel_delta_v);

		pixel_samples_scale = 1.0/samples_per_pixel;
  }

	point sample_square() const {
		return point(random_double() - 0.5, random_double() - 0.5, 0);
	}

	Ray<double> get_ray_sample(int i, int j) const {
		point offset = sample_square();
		point pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);
		point ray_direction = pixel_sample - camera_center;
		return Ray(camera_center, ray_direction);
	}

  color ray_color(const Ray<double> &r, const Hittable &world) {
    HitRecord rec;

    if (world.hit(r, Interval<double>(0.0, infinity), rec)) {
      return 0.5 * (rec.normal + color(1.0, 1.0, 1.0));
    }

    point unit_direction = unit_vector(r.direction());
    double a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(1.0, 0.1, 0.1);
  }
};

#endif
