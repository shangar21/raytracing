#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "ray.h"

class Sphere: public Hittable {
	public:
		double r;
		point c;
		
		Sphere(point center, double radius) : c(center), r(radius) {} 

		bool hit(const Ray<double>& r, double ray_tmin, double ray_tmax, HitRecord& rec) const {
			point oc = c - r.origin();
			double a = r.direction().length_squared();
			double b = -2.0 * dot(r.direction(), oc);
			double c = oc.length_squared() - (radius * radius);
			double discriminant = b*b - 4*a*c;

			if (discriminant >= 0) {
				double t1 = (-b - std::sqrt(discriminant)) / (2.0 * a);
				double t2 = (-b + std::sqrt(discriminant)) / (2.0 * a);

				if ((ray_tmin <= t1 && t1 <= ray_tmax) || (ray_tmin <= t2 && t2 <= ray_tmax)){
					double t = (ray_tmin <= t1 && t1 <= ray_tmax) ? t1 : t2;
					point p = r.at(t);
					point normal = (p - c) / r;
					rec.t = t;
					rec.p = p;
					rec.set_face_normal(r, normal);
					return true;
				}
			}

			return false;
		}
	
};

#endif
