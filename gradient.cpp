// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

using namespace cv;
using namespace std;

void getGradient(
	cv::Mat &input, 
	cv::Mat &blurredOutput);

float radian_conversion = 180/M_PI;

// int main( int argc, char** argv ) {

// // LOADING THE IMAGE
// char* imageName = argv[1];

// Mat image;
// image = imread( imageName, 1 );

// if( argc != 2 || !image.data ) {
//     printf( " No image data \n " );
//     return -1;
//     }

//     // CONVERT COLOUR, BLUR
//     Mat gray_image;
//     cvtColor( image, gray_image, CV_BGR2GRAY );
//     blur(gray_image, gray_image, Size(3,3));

//     Mat grad_dir;
//     getGradient(gray_image, grad_dir);

//     imwrite("grad_dir.jpg", grad_dir);

// return 0;
// }

void getGradient(cv::Mat &input, cv::Mat &output)
{
    Mat dx, dy;
    Sobel(input, dx, CV_32F, 1, 0);
    Sobel(input, dy, CV_32F, 0, 1);

    imwrite("dx.jpg", dx);
    imwrite("dy.jpg", dy);

    Mat mag;
    cartToPolar(dx, dy, mag, output); // output is assigned to angle
}