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
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

// Ground truth faces array
Rect dart4[] = {Rect(334, 102, 151, 166)};
Rect dart5[] = {Rect(52, 139, 76, 73), Rect(45, 245, 79, 80), Rect(186, 200, 64, 85), Rect(250, 167, 56, 65),
								Rect(294, 239, 56, 74), Rect(372, 185, 73, 69), Rect(429, 234, 53, 67), Rect(518, 177, 48, 64),
								Rect(559, 246, 58, 67), Rect(646, 184, 59, 67), Rect(681, 242, 50, 71)};
Rect dart6[] = {Rect(287, 116, 39, 42)};
Rect dart7[] = {Rect(349, 186, 68, 95)};
Rect dart9[] = {Rect(85, 206, 114, 140)};
Rect dart11[] = {Rect(320, 80, 67, 69)};
Rect dart13[] = {Rect(421, 125, 110, 129)};
Rect dart14[] = {Rect(467, 220, 81, 99), Rect(723, 188, 101, 101)};

/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

  // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ ) {
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	// for ( int i = 0; i < sizeof(dart14)/sizeof(dart14[0]); i++) {
	// 	rectangle(frame, Point(dart14[i].x, dart14[i].y), Point(dart14[i].x + dart14[i].width, dart14[i].y + dart14[i].height), Scalar( 0, 0, 255 ), 2);
	// }

}
