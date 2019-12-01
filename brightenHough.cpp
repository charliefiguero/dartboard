/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - brightenHough.cpp
//
/////////////////////////////////////////////////////////////////////////////
#include "hough.h"

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
#include <unordered_map>
#include <math.h>

using namespace std;
using namespace cv;

// -------------------------------------Main-------------------------------------
/** @function main */
int main( int argc, const char** argv ) {
	Mat frame = imread(argv[1]);
	cvtColor( frame, frame, CV_BGR2GRAY );
    Mat output = frame.clone(); 

    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            if (frame.at<float>(i,j) != 0) output.at<float>(i,j) = 255;
        }
    }

    imwrite("brightenedHough.jpg", output);

	return 0;
}
// ------------------------------------------------------------------------------
