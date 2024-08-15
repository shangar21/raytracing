#include <iostream>
#include <opencv2/opencv.hpp>

// External CUDA function
extern "C" void render_image(double* h_R, double* h_G, double* h_B, int image_width, int image_height, 
    double pixel00_x, double pixel00_y, double pixel00_z, 
    double pixel_delta_u_x, double pixel_delta_u_y, double pixel_delta_u_z, 
    double pixel_delta_v_x, double pixel_delta_v_y, double pixel_delta_v_z, 
    double radius, double sphere_o_x, double sphere_o_y, double sphere_o_z);

int main() {
    const int image_width = 1920;
    const int image_height = 1080;
    const int num_pixels = image_width * image_height;

    // Host memory for RGB components
    double *h_R = new double[num_pixels];
    double *h_G = new double[num_pixels];
    double *h_B = new double[num_pixels];

    // Define camera parameters
		double fl = 1.0;
		double v_h = 2.0;
		double v_w = v_h * ((double)image_width / image_height);
		
		double pixel_delta_u_x = v_w / (double) image_width;
		double pixel_delta_u_y = 0.0;
		double pixel_delta_u_z = 0.0;

		double pixel_delta_v_x = 0.0;
		double pixel_delta_v_y = -v_h / (double) image_height;
		double pixel_delta_v_z = 0.0;

		double pixel00_x = 0.0 - v_w / 2.0;
		double pixel00_y = 0.0 + v_h / 2.0;
		double pixel00_z = -1.0;

		double radius = 0.5;
		double sphere_o_x = 0.0;
		double sphere_o_y = 0.0;
		double sphere_o_z = -1.0;

    // Call the CUDA function to render the image
    render_image(h_R, h_G, h_B, image_width, image_height,
                 pixel00_x, pixel00_y, pixel00_z,
                 pixel_delta_u_x, pixel_delta_u_y, pixel_delta_u_z,
                 pixel_delta_v_x, pixel_delta_v_y, pixel_delta_v_z,
                 radius, sphere_o_x, sphere_o_y, sphere_o_z);

    // Convert the result to an OpenCV Mat and save it as a PNG
    cv::Mat image(image_height, image_width, CV_8UC3);
    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            int idx = j * image_width + i;
            image.at<cv::Vec3b>(j, i)[0] = static_cast<unsigned char>(h_B[idx] * 255.99); // Blue channel
            image.at<cv::Vec3b>(j, i)[1] = static_cast<unsigned char>(h_G[idx] * 255.99); // Green channel
            image.at<cv::Vec3b>(j, i)[2] = static_cast<unsigned char>(h_R[idx] * 255.99); // Red channel
        }
    }

    cv::imwrite("rendered_image.png", image);

    // Cleanup
    delete[] h_R;
    delete[] h_G;
    delete[] h_B;

    return 0;
}

