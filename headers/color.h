#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using color = vec3<double>;

void writeColor(std::ostream &out, const color pixel) {
  double r = pixel.x();
  double g = pixel.y();
  double b = pixel.z();

  int rbyte = int(255.99 * r);
  int gbyte = int(255.99 * g);
  int bbyte = int(255.99 * b);

  out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

void writeColorOpenCV(Eigen::MatrixXd R, Eigen::MatrixXd G, Eigen::MatrixXd B, int h, int w){
	cv::Mat R_mat(h, w, CV_64F, R.data());
	cv::Mat G_mat(h, w, CV_64F, G.data());
	cv::Mat B_mat(h, w, CV_64F, B.data());

	R_mat *= 255.99; G_mat *= 255.99; B_mat *= 255.99;

	R_mat.convertTo(R_mat, CV_8U);
	G_mat.convertTo(G_mat, CV_8U);
	B_mat.convertTo(B_mat, CV_8U);

	std::cout << R_mat;

	std::vector<cv::Mat> channels = {B_mat, G_mat, R_mat};
	cv::Mat img;
	cv::merge(channels, img);

	cv::imwrite("img.jpg", R_mat);
}

#endif
