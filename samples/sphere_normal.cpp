
#include "color.h"
#include "ray.h"
#include "vec3.h"

#include <iostream>

double hit_sphere(vec3<double> center, double radius, ray<double>& r){
	vec3<double> oc = center - r.origin();
	double a = dot(r.direction(), r.direction());
	double b = -2.0 * dot(r.direction(), oc);
	double c = dot(oc, oc) - radius * radius;
	double discriminant = b*b - 4*a*c;

	if (discriminant < 0) {
		return -1.0;
	}else{
		return (-b - std::sqrt(discriminant)) / (2.0 * a);
	}
}

vec3<double> ray_color(ray<double>& r){
	double t = hit_sphere(vec3<double>(0.0, 0.0, -1.0), 0.5, r);

	if  (t > 0.0){
		vec3<double> normal = unit_vector(r.at(t) - vec3<double>(0, 0, -1));
		return 0.5 * vec3<double>(normal.x() + 1, normal.y() + 1, normal.z() + 1);
	}

	vec3<double> unit_direction = unit_vector(r.direction());
	double a = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - a) * vec3<double>(1.0, 1.0, 1.0) + a * vec3<double>(0.0, 0.0, 1.0);
}

int main() {
	double aspect_ratio = 16.0/9.0;
	int image_width = 1920;

	int image_height = int(image_width / aspect_ratio);
	image_height = (image_height < 1) ? 1 : image_height;

	double fl = 1.0;
	double v_h = 2.0;
	double v_w = v_h * ((double)image_width / image_height);
	vec3<double> camera_center = vec3<double>(0.0, 0.0, 0.0);
	
	vec3<double> viewport_u = vec3<double>(v_w, 0.0, 0.0);
	vec3<double> viewport_v = vec3<double>(0.0, -v_h, 0.0);

	vec3<double> pixel_delta_u = viewport_u / (double)image_width;
	vec3<double> pixel_delta_v = viewport_v / (double)image_height;

	vec3<double> viewport_ul = camera_center - vec3<double>(0, 0, fl) - viewport_u / 2.0 - viewport_v / 2.0;
	vec3<double> pixel_00 = viewport_ul + 0.5 * (pixel_delta_u + pixel_delta_v);

	std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

	for (int j = 0; j < image_height; j++){
		std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
		for(int i = 0; i < image_width; i++){
			vec3<double> pixel_center = pixel_00 + ((double)i * pixel_delta_u) + ((double)j * pixel_delta_v);
			vec3<double> ray_dir = pixel_center - camera_center;
			ray r = ray(camera_center, ray_dir);
			vec3<double> pixel = ray_color(r);
			writeColor(std::cout, pixel);
		}
	}
	std::clog << "\rDone :)" << std::flush;
}
