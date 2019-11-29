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
// 	Mat canny_img;

// 	getCanny(frame, canny_img, 50, 200);
// 	hough_lines(frame, canny_img, 140);
// 	hough_circles(frame, canny_img, 15, 10, 300);

// 	cout << "hough shit" << endl;
// 	return 0;
// }

// Performs hough transform on an image with a given threshold
void hough_lines(Mat img, Mat canny_img, int thresholdHough, Rect focus_area) {

	int max_rho = hypot(img.rows, img.cols);
	Mat hough_img(2*max_rho, 180, CV_32FC1); // This img contains hough space
  	Mat lines_img = img.clone(); // Detected lines are drawn to this img

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
}

void hough_circles(Mat img, Mat canny_img, int thresholdVal, int minRadius, int maxRadius, Rect focus_area) {
	Mat directionImg, magnitudeImg;
	getGradient(img, directionImg, magnitudeImg);
	Mat circles_img = img.clone();

	// for (int i = 0; i < img.rows; i++) {
	// 	for (int j = 0; j < img.cols; j++) {
	// 		cout << directionImg.at<float>(i, j) << endl;
	// 	}
	// }

	int size[3] = {img.rows + (2 * maxRadius), img.cols + (2 * maxRadius), maxRadius - minRadius + 1}; // adding border padding of maxRadius
	Mat circle_centres(3, size, CV_32F, Scalar(0));

	Mat flattenedHough((img.rows + (2 * maxRadius)), img.cols + (2 * maxRadius), CV_32FC1);
	
	// Creates houghspace for image on hough_img
	for ( int i = 0; i < canny_img.rows; i++ ) {
		for( int j = 0; j < canny_img.cols; j++ ) { //loop over pixels in img
			if ((int) canny_img.at<uchar>(i, j) == 255) { // if canny detects an edge here

				for (int r = minRadius; r <= maxRadius; r++) {
					float direction = directionImg.at<float>(i, j);
					int x0Pos = j + (r * cos(direction)) + maxRadius; // maxRadius to account for border padding
					int x0Neg = j - (r * cos(direction)) + maxRadius;
					int y0Pos = i + (r * sin(direction)) + maxRadius;
					int y0Neg = i - (r * sin(direction)) + maxRadius;
					//if (y0Neg > img.cols + (2 * maxRadius)) cout << "x: " << x0Neg << endl;
					circle_centres.at<float>(y0Pos, x0Pos, (r - minRadius))++; // indexing is offset by the minimum radius
					circle_centres.at<float>(y0Neg, x0Neg, (r - minRadius))++;	
					flattenedHough.at<float>(y0Pos, x0Pos)++;	
					flattenedHough.at<float>(y0Neg, x0Neg)++;	
				}
			}
		}
	}

	// Mat hough_img(2*max_rho, 180, CV_32FC1);

	imwrite("hough_circles.jpg", circle_centres);
	imwrite("hough_space.jpg", flattenedHough);

	for ( int i = 0; i < img.rows + (2 * maxRadius); i++ ) {
		for( int j = 0; j < img.cols + (2 * maxRadius); j++ ) { 
			for (int r = minRadius; r <= maxRadius; r++) {
				if (circle_centres.at<float>(i, j, r - minRadius) > thresholdVal) {
					Point centre = Point(j - maxRadius, i - maxRadius);
					circle(circles_img, centre, r, Scalar( 0, 255, 0 ));
				}
				
			}
			
		}
	}	
	imwrite("circles.jpg", circles_img);		

}

void getCanny( Mat &inputImg, Mat &output, int lowThreshold, int highThreshold) {
	Mat gray_img, canny_img;

	// Create grey scale image
	cvtColor( inputImg, gray_img, CV_BGR2GRAY );
	blur( gray_img, canny_img, Size(3,3) ); // Blur image for more effective sobel convolution
	Canny( canny_img, output, lowThreshold, highThreshold, 3 ); 	// resulting image gives pixel values on prominent lines as 255 and other pixels as 0

	imwrite("canny.jpg", output);
}

void getGradient(Mat &input, Mat &directionOutput, Mat &magnitudeOutput)
{
    Mat dx, dy, gray_img;
	cvtColor( input, gray_img, CV_BGR2GRAY );

    Sobel(gray_img, dx, CV_32F, 1, 0);
    Sobel(gray_img, dy, CV_32F, 0, 1);

    imwrite("dx.jpg", dx);
    imwrite("dy.jpg", dy);

    cartToPolar(dx, dy, magnitudeOutput, directionOutput); // output is assigned to angle
}
