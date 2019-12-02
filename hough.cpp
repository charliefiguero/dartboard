//"We cannot get out. We cannot get out. They have taken the bridge and Second Hall. Frár and Lóni and Náli fell there bravely while the rest retreated to Mazarbul.
//We still hold the chamber but hope is fading now. Óin’s party went five days ago but today only four returned.
//The pool is up to the wall at West-gate. The Watcher in the Water took Óin -- we cannot get out. The end comes soon. We hear drums, drums in the deep."
//They are coming.

/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - hough.cpp
//
/////////////////////////////////////////////////////////////////////////////
#include "hough.h"

/** Global variables */
const float radian_conversion = M_PI / (float) 180;

// -------------------------------------Main-------------------------------------
/** @function main */
// int main( int argc, const char** argv ) {
// 	// Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
// 	Mat frame = imread(argv[1]);
// 	Mat canny_img;
//  Mat draw_image = frame.clone(); 

// 	getCanny(frame, canny_img, 50, 200);
// 	hough_lines(frame, canny_img, 140);
// 	hough_circles(frame, canny_img, 15, 10, 300);

// 	cout << "hough shit" << endl;
// 	return 0;
// }
// ------------------------------------------------------------------------------

// Performs hough transform on an image with a given threshold
int hough_lines(Mat canny_img, Rect focus_area, Mat direction_image, Mat lines_img, int thresholdHough) {
	int angles_per_degree = 2;
	int delta_theta = 20; // 360 to avoid negative numbers
	int line_count = 0;

	int max_rho = hypot(lines_img.rows, lines_img.cols);
	Mat hough_img(2*max_rho, 180*angles_per_degree, CV_32FC1); // This img contains hough space
	hough_img = Scalar(0);

	int xStart = focus_area.x;
	int xEnd = focus_area.x + focus_area.width;
	int yStart = focus_area.y;
	int yEnd = focus_area.y + focus_area.height;

	if( xStart < 0 ) xStart = 0;
	if( yStart < 0 ) yStart = 0;
	if( xEnd > lines_img.cols ) xEnd = lines_img.cols;
	if( yEnd > lines_img.rows ) yEnd = lines_img.rows;

    // Creates houghspace for image on hough_img
	for ( int i = yStart; i < yEnd; i++ ) {
		for( int j = xStart; j < xEnd; j++ ) { //loop over pixels in img

			if ((int) canny_img.at<uchar>(i, j) == 255) { // if canny detects an edge here

				// (gradient direction + PI to find edge direction) -> radian conversion -> -delta_theta for start of angle range -> mult by angles_per_degree
				// int theta_start = (( round(direction_image.at<float> (i, j) - M_PI/2) / radian_conversion) - delta_theta) * angles_per_degree;
				// int theta_end = (( round(direction_image.at<float> (i, j) - M_PI/2) / radian_conversion) + delta_theta) * angles_per_degree;
				int theta_start = 0;
				int theta_end = 360;

				for (int theta = theta_start; theta < theta_end; theta++) { // Increment pixel value in hough space
					float correct_theta_rads = ((theta%(angles_per_degree*180))/angles_per_degree) * radian_conversion; 
					int rho = round((j * cos(correct_theta_rads)) + (i * sin(correct_theta_rads))) + max_rho; // x.cos(theta) + y.sin(theta)
					// hough_img.at<float>(rho, theta%(angles_per_degree*180))++; // y then x-- therefore theta is plotted along x
					hough_img.at<float>(rho, theta%(angles_per_degree*180))++;
				}
			}
		}
	}
	// hough space has been created
	imwrite("result_hough_space.jpg", hough_img);
	//rectangle(lines_img, Point(focus_area.x, focus_area.y), Point(focus_area.x + focus_area.width, focus_area.y + focus_area.height), Scalar( 255, 255, 255 ), 2);

	// Go through every pixel in hough
	// If above threshhold add to pixel in corresponding houghspace
	for ( int i = 0; i < hough_img.rows; i++ ) {
		for( int j = 0; j < hough_img.cols; j++ ) {
			if (hough_img.at<float>(i, j) > thresholdHough) {

			  	float rho = (i - max_rho), theta = (j/angles_per_degree)*radian_conversion;
			  	Point pt1, pt2; // pt1 is x intercept, pt2 is y

			  	double a = cos(theta), b = sin(theta);
			  	double x0 = a*rho, y0 = b*rho;
				
			  	pt1.x = cvRound(x0 + 2000*(-b));
			  	pt1.y = cvRound(y0 + 2000*(a));
			  	pt2.x = cvRound(x0 - 2000*(-b));
			  	pt2.y = cvRound(y0 - 2000*(a));

				line( lines_img, pt1, pt2, Scalar(0,0,255), 3, CV_AA); // draws lines
				line_count++;	
			}

		}
	}
	return line_count;
}

void create_circle_houghspace(Mat img, Mat canny_img, Mat &circle_hough_space, Mat flattened_circle_hough, Mat directionImg, int minRadius, int maxRadius, Rect focus_area) {
	int xStart = focus_area.x;
	int xEnd = focus_area.x + focus_area.width;
	int yStart = focus_area.y;
	int yEnd = focus_area.y + focus_area.height;
	
	// Creates houghspace for image on hough_img
	for ( int i = yStart; i < yEnd; i++ ) {
		for( int j = xStart; j < xEnd; j++ ) { //loop over pixels in img

			if ((int) canny_img.at<uchar>(i, j) == 255) { 				// if canny detects an edge here
				for (int r = minRadius; r <= maxRadius; r++) {
					float direction = directionImg.at<float>(i, j);
					int x0Pos = j + (r * cos(direction)) + maxRadius;   // finds circle centres for edge
					int x0Neg = j - (r * cos(direction)) + maxRadius;
					int y0Pos = i + (r * sin(direction)) + maxRadius;
					int y0Neg = i - (r * sin(direction)) + maxRadius;

					circle_hough_space.at<float>(y0Pos, x0Pos, (r - minRadius))++; // indexing is offset by the minimum radius
					circle_hough_space.at<float>(y0Neg, x0Neg, (r - minRadius))++;	
					//cout << circle_hough_space.at<float>(y0Neg, x0Neg, (r - minRadius)) << endl;
					flattened_circle_hough.at<float>(y0Pos, x0Pos)++;	
					flattened_circle_hough.at<float>(y0Neg, x0Neg)++;	
				}
			}
		}
	}

	imwrite("hough_space.jpg", flattened_circle_hough);
}

void draw_circles( Mat circles_img, Mat circle_hough_space, Mat flattened_circle_hough, vector<Point3d> &circle_centres, int thresholdVal, int minRadius, int maxRadius) {
	// draws circle onto surcools image and gathers surcoools count
	for ( int i = 0; i < flattened_circle_hough.rows; i++ ) {
		for( int j = 0; j < flattened_circle_hough.cols; j++ ) { 
			
			for (int r = minRadius; r <= maxRadius; r++) {
				
				if (circle_hough_space.at<float>(i, j, r - minRadius) > thresholdVal) { // -minRadius to offset padding 
					Point centre = Point(j - maxRadius, i - maxRadius); // -maxRadius to offset padding 
					circle(circles_img, centre, r, Scalar( 0, 255, 0 ));
					circle_centres.push_back(Point3d(j - maxRadius, i - maxRadius, r));
				}
			}
		}
	}			
}

int find_circle_centres(Mat flattened_hough_space, int threshold_val, vector<Point> &circle_centres) {
	int circleCount = 0;

	for ( int i = 0; i < flattened_hough_space.rows; i++ ) {
		for( int j = 0; j < flattened_hough_space.cols; j++ ) { 
			
			 if (flattened_hough_space.at<float>(i, j) > threshold_val) {
			 	circle_centres.push_back(Point(j, i));
			 	circleCount++;
			 }
		}
	}
	return circleCount;
}

void getCanny( Mat &inputImg, Mat &output, int lowThreshold, int highThreshold) {
	Mat gray_img, canny_img;

	// Create grey scale image
	cvtColor( inputImg, gray_img, CV_BGR2GRAY );
	blur( gray_img, canny_img, Size(3,3) ); // Blur image for more effective sobel convolution
	Canny( canny_img, output, lowThreshold, highThreshold, 3 ); 	// resulting image gives pixel values on prominent lines as 255 and other pixels as 0

	//imwrite("canny.jpg", output);
}

void getGradient(Mat &input, Mat &directionOutput, Mat &magnitudeOutput)
{
    Mat dx, dy, gray_img;
	cvtColor( input, gray_img, CV_BGR2GRAY );

    Sobel(gray_img, dx, CV_32F, 1, 0);
    Sobel(gray_img, dy, CV_32F, 0, 1);

    //imwrite("dx.jpg", dx);
    //imwrite("dy.jpg", dy);

    cartToPolar(dx, dy, magnitudeOutput, directionOutput); // output is assigned to angle
}