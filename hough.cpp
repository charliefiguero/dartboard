/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <unordered_map>
#include <math.h>

using namespace std;
using namespace cv;

/** Function Headers */
void hough_transform(Mat img, int thresholdHough, int lowThresholdCanny, int highThresholdCanny);

/** Global variables */
float radian_conversion = M_PI / (float) 180;

/** @function main */
int main( int argc, const char** argv ) {
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	hough_transform(frame, 200, 50, 200);
	return 0;
}

// Performs hough transform on an image with a given threshold
void hough_transform(Mat img, int thresholdHough, int lowThresholdCanny, int highThresholdCanny) {

	Mat gray_img, canny_img;
	Mat hough_img(1236, 180, CV_32FC1); // This img contains hough space
  Mat lines_img = img.clone(); // Detected lines are drawn to this img

	// Create grey scale image
	cvtColor( img, gray_img, CV_BGR2GRAY );

	blur( gray_img, canny_img, Size(3,3) ); // Blur image for more effective sobel convolution
	Canny( canny_img, canny_img, lowThresholdCanny, highThresholdCanny, 3 ); 	// resulting image gives pixel values on prominent lines as 255 and other pixels as 0

  imwrite("canny_img.jpg", canny_img); // GOOD
	for ( int i = 0; i < canny_img.rows; i++ ) {
		for( int j = 0; j < canny_img.cols; j++ ) { //loop over pixels in img
			std::cout << hough_img.at<float>(i,j);
		}
	}


  // Creates houghspace for image on hough_img
	for ( int i = 0; i < canny_img.rows; i++ ) {
		for( int j = 0; j < canny_img.cols; j++ ) { //loop over pixels in img

			if (canny_img.at<uchar>(i, j) == 255) { // if canny detects an edge here

				for (int theta = 0; theta < 180; theta++) { // Increment pixel value in hough space
					int rho = abs((j * cos( theta * radian_conversion ) ) + (i * sin( theta * radian_conversion))); // x.cos(theta) + y.sin(theta)
					hough_img.at<float>(rho, theta) += 1; // y then x-- therefore theta is plotted along x
				}
			}
		}
	}


	// Draws lines detected in hough space onto image
	for ( int i = 0; i < hough_img.rows; i++ ) {
		for( int j = 0; j < hough_img.cols; j++ ) {
			if (hough_img.at<float>(i, j) > thresholdHough) {
				float rho = i, theta = j;
			  Point pt1, pt2;
			  double a = cos(theta), b = sin(theta);
			  double x0 = a*rho, y0 = b*rho;
			  pt1.x = cvRound(x0 + 1000*(-b));
			  pt1.y = cvRound(y0 + 1000*(a));
			  pt2.x = cvRound(x0 - 1000*(-b));
			  pt2.y = cvRound(y0 - 1000*(a));
			  line( lines_img, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
			}
		}
	}

}
