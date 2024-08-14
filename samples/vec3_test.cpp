#include "vec3.h"

int main() {
    vec3<double> v1(1.0, 2.0, 3.0);
    vec3<double> v2(4.0, 5.0, 6.0);

    vec3<double> v3 = v1 + v2;
    vec3<double> v4 = v1 - v2;
    vec3<double> v5 = v1 * v2;
    vec3<double> v6 = v1 / v2;
    vec3<double> v7 = 2.0 * v1;
    vec3<double> v8 = v1 * 2.0;
    vec3<double> v9 = v1 / 2.0;

    double dot_product = dot(v1, v2);
    vec3<double> cross_product = cross(v1, v2);

    std::cout << "v3 (v1 + v2): " << v3 << std::endl;
    std::cout << "v4 (v1 - v2): " << v4 << std::endl;
    std::cout << "v5 (v1 * v2): " << v5 << std::endl;
    std::cout << "v6 (v1 / v2): " << v6 << std::endl;
    std::cout << "v7 (2.0 * v1): " << v7 << std::endl;
    std::cout << "v8 (v1 * 2.0): " << v8 << std::endl;
    std::cout << "v9 (v1 / 2.0): " << v9 << std::endl;
    std::cout << "dot(v1, v2): " << dot_product << std::endl;
    std::cout << "cross(v1, v2): " << cross_product << std::endl;

    return 0;
}
