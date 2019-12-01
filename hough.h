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
int hough_lines(Mat canny_img, Rect focus_area, Mat direction_image, Mat lines_img, int thresholdHough);
void create_circle_houghspace(Mat img, Mat canny_img, Mat &circle_hough_space, Mat flattened_circle_hough, Mat directionImg, int minRadius, int maxRadius, Rect focus_area);

void draw_circles( Mat circles_img, Mat circle_hough_space, Mat flattened_circle_hough, vector<Point3d> &circle_centres, int thresholdVal, int minRadius, int maxRadius);
int find_circle_centres(Mat flattened_hough_space, int threshold_val, vector<Point> &circle_centres);

void getGradient(Mat &input, Mat &directionOutput, Mat &magnitudeOutput);
void getCanny( Mat &inputImg, Mat &output, int lowThreshold, int highThreshold); 

