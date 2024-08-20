#include <iostream>
#include <opencv2/opencv.hpp>

extern "C" void render_image(double *h_R, double *h_G, double *h_B,
                             int image_width, int image_height);

int main() {
  const int image_width = 1920;
  const int image_height = 1080;

  double *h_R = new double[image_width * image_height];
  double *h_G = new double[image_width * image_height];
  double *h_B = new double[image_width * image_height];

  render_image(h_R, h_G, h_B, image_width, image_height);

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
