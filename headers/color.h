#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using color = vec3<double>;

inline double linear_to_gamma(double linear_component) {
  if (linear_component > 0)
    return std::sqrt(linear_component);
  return 0;
}

void writeColor(std::ostream &out, const color pixel) {
  double r = pixel.x();
  double g = pixel.y();
  double b = pixel.z();

  int rbyte = int(255.99 * r);
  int gbyte = int(255.99 * g);
  int bbyte = int(255.99 * b);

  out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

void writeColorOpenCV(cv::Mat &R_mat, cv::Mat &G_mat, cv::Mat &B_mat) {
  // Scale the color channels from [0, 1] to [0, 255]
  R_mat *= 255.99;
  G_mat *= 255.99;
  B_mat *= 255.99;

  // Convert to 8-bit unsigned integer (CV_8U)
  R_mat.convertTo(R_mat, CV_8U);
  G_mat.convertTo(G_mat, CV_8U);
  B_mat.convertTo(B_mat, CV_8U);

  // Merge the channels into a single BGR image
  std::vector<cv::Mat> channels = {B_mat, G_mat,
                                   R_mat}; // OpenCV uses BGR order by default
  cv::Mat img;
  cv::merge(channels, img);

  // Write the image to a file
  cv::imwrite("img.png", img);
}

#endif
