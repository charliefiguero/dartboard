/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////
#include "hough.h"

/** Global variables */
float radian_conversion = M_PI / (float) 180;

/** @function main */
// int main( int argc, const char** argv ) {
// 	// Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
// 	Mat frame = imread(argv[1]);

// 	hough_transform(frame, 140, 50, 200);

// 	cout << "hello" << endl;
// 	return 0;
// }

// Performs hough transform on an image with a given threshold
void hough_transform(Mat img, int thresholdHough, int lowThresholdCanny, int highThresholdCanny) {

	int max_rho = hypot(img.rows, img.cols);
	Mat gray_img, canny_img;
	Mat hough_img(2*max_rho, 180, CV_32FC1); // This img contains hough space
  	Mat lines_img = img.clone(); // Detected lines are drawn to this img

	// Create grey scale image
	//cvtColor( img, gray_img, CV_BGR2GRAY );
	blur( gray_img, canny_img, Size(3,3) ); // Blur image for more effective sobel convolution
	Canny( img, canny_img, lowThresholdCanny, highThresholdCanny, 3 ); 	// resulting image gives pixel values on prominent lines as 255 and other pixels as 0

	//"We cannot get out. We cannot get out. They have taken the bridge and Second Hall. Frár and Lóni and Náli fell there bravely while the rest retreated to Mazarbul.
	//We still hold the chamber but hope is fading now. Óin’s party went five days ago but today only four returned.
	//The pool is up to the wall at West-gate. The Watcher in the Water took Óin -- we cannot get out. The end comes soon. We hear drums, drums in the deep."
  	//They are coming.

  // Creates houghspace for image on hough_img
	for ( int i = 0; i < canny_img.rows; i++ ) {
		for( int j = 0; j < canny_img.cols; j++ ) { //loop over pixels in img

			if ((int) canny_img.at<uchar>(i, j) == 255) { // if canny detects an edge here

				for (int theta = 0; theta < 180; theta++) { // Increment pixel value in hough space
					int rho = round((j * cos( (theta) * radian_conversion ) ) + (i * sin( (theta) * radian_conversion))) + max_rho; // x.cos(theta) + y.sin(theta)
					hough_img.at<float>(rho, theta) += 1; // y then x-- therefore theta is plotted along x
				}
			}
		}
	}

	imwrite("result_hough_space.jpg", hough_img);

	// Go through every pixel in hough
	// If above threshhold add to pixel in corresponding houghspace

	for ( int i = 0; i < hough_img.rows; i++ ) {
		for( int j = 0; j < hough_img.cols; j++ ) {
			if (hough_img.at<float>(i, j) > thresholdHough) {
			  	float rho = (i - max_rho), theta = (j)*radian_conversion;
			  	Point pt1, pt2; // pt1 is x intercept, pt2 is y

			  	double a = cos(theta), b = sin(theta);
			  	double x0 = a*rho, y0 = b*rho;
				
			  	pt1.x = cvRound(x0 + 2000*(-b));
			  	pt1.y = cvRound(y0 + 2000*(a));
			  	pt2.x = cvRound(x0 - 2000*(-b));
			  	pt2.y = cvRound(y0 - 2000*(a));

			  	line( lines_img, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
			}
		}
	}
	
	imwrite("hough_result.jpg", lines_img);
	imwrite("canny.jpg", canny_img);

}
