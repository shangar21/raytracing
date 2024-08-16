#include "camera.cu"
#include <cuda_runtime.h>

int main() {
  const int image_width = 1920;
  const int image_height = 1080;

  // Define pixel00, pixel_delta_u, and pixel_delta_v using double3
  double fl = 1.0;
  double v_h = 2.0;
  double v_w = v_h * ((double)image_width / image_height);
  double3 cam_center = make_double3(0.0, 0.0, 0.0);
  double3 viewport_u = make_double3(v_w, 0.0, 0.0);
  double3 viewport_v = make_double3(0.0, -v_h, 0.0);

  double3 pixel_delta_u = viewport_u / (double)image_width;
  double3 pixel_delta_v = viewport_v / (double)image_width;

  double3 pixel00 = cam_center - make_double3(0.0, 0.0, fl) - viewport_u / 2.0 -
                    viewport_v / 2.0;

  double radius = 0.5;
  double3 sphere_o = make_double3(0, 0, -1);

  // Allocate memory for the color components
  double *h_R = new double[image_width * image_height];
  double *h_G = new double[image_width * image_height];
  double *h_B = new double[image_width * image_height];

  // Call the render_image function using double3 arguments
  render_image(h_R, h_G, h_B, image_width, image_height, pixel00, pixel_delta_u,
               pixel_delta_v, radius, sphere_o);

  // Convert the result to an OpenCV Mat and save it as a PNG
  cv::Mat image = cv::Mat(image_height, image_width, CV_8UC3);

  for (int j = 0; j < image_height; ++j) {
    for (int i = 0; i < image_width; ++i) {
      int idx = j * image_width + i;
      image.at<cv::Vec3b>(j, i)[0] =
          static_cast<unsigned char>(h_B[idx] * 255.99); // Blue channel
      image.at<cv::Vec3b>(j, i)[1] =
          static_cast<unsigned char>(h_G[idx] * 255.99); // Green channel
      image.at<cv::Vec3b>(j, i)[2] =
          static_cast<unsigned char>(h_R[idx] * 255.99); // Red channel
    }
  }

  cv::imwrite("rendered_image.png", image);

  // Cleanup
  delete[] h_R;
  delete[] h_G;
  delete[] h_B;

  return 0;
}
