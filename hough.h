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
void hough_lines(Mat img, Mat canny_img, Mat lines_img, int thresholdHough, Rect focus_area);
int hough_circles(Mat img, Mat canny_img, Mat circles_img, Mat directionImg, int thresholdVal, int minRadius, int maxRadius, Rect focus_area);
void getGradient(Mat &input, Mat &directionOutput, Mat &magnitudeOutput);
void getCanny( Mat &inputImg, Mat &output, int lowThreshold, int highThreshold); 

