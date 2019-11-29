/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
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

/** Function Headers */
int detectAndDisplay( Mat frame );
float calculateIOU(Rect detectedRectangle, Rect groundTruthRectangle);
void findCorrectIOU(int i, int numberOfFaces);
float calculateTpr();
float calculateF1Score(int numberOfFaces);
void hough_lines(Mat img, int thresholdHough, int lowThresholdCanny, int highThresholdCanny);
void violaJones(Mat frame, vector<Rect> &faces);

// Ground truth faces array -------------------------------------------------------------------------
Rect dart0_ground[] = {Rect(449, 16, 151, 177)};
Rect dart1_ground[] = {Rect(196, 128, 198, 190)};
Rect dart2_ground[] = {Rect(106, 96, 86, 87)};
Rect dart3_ground[] = {Rect(330, 150, 60, 66)};
Rect dart4_ground[] = {Rect(202, 94, 183, 205)};
Rect dart5_ground[] = {Rect(434, 143, 97, 11)};
Rect dart6_ground[] = {Rect(214, 119, 62, 62)};
Rect dart7_ground[] = {Rect(253, 172, 144, 144)};
Rect dart8_ground[] = {Rect(64, 252, 67, 88), Rect(850, 216, 103, 120)};
Rect dart9_ground[] = {Rect(203, 61, 231, 217)};
Rect dart10_ground[] = {Rect(96, 105, 91, 109), Rect(585, 128, 56, 84), Rect(919, 148, 32, 66)};
Rect dart11_ground[] = {Rect(175, 105, 62, 67)};
Rect dart12_ground[] = {Rect(150, 76, 68, 139)};
Rect dart13_ground[] = {Rect(273, 128, 129, 124)};
Rect dart14_ground[] = {Rect(121, 103, 125, 124), Rect(989, 96, 120, 123)};
Rect dart15_ground[] = {Rect(151, 59, 138, 139)};
// --------------------------------------------------------------------------------------------------

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

// these must be changed when using a different file
// string saveImageLocation = "report/dart15_detected.jpg";
int lengthGT = sizeof(dart14_ground)/sizeof(dart14_ground[0]);
Rect GTArray[sizeof(dart14_ground)/sizeof(dart14_ground[0])] = dart14_ground;
float thresholdForCalculations = 0.45;

// key is the index of the face that has been chosen
// value is the index of the GT which chose the face
std::unordered_map<int, int> chosenFaces;
// array to store the correct IOUs for each GT
float GT_IOU_values[sizeof(GTArray)/sizeof(GTArray[0])];
float ious[20][20];

/** @function main */
int main( int argc, const char** argv ) {
  	// 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	int numberOfFaces = detectAndDisplay( frame );

	// make sure each GT has only 1 associated face
	for (int i = 0; i < lengthGT; i++) {
		findCorrectIOU(i, numberOfFaces);
	}

	// find TPR
	float tpr = calculateTpr();
	float f1Score = calculateF1Score(numberOfFaces);
	//std::cout << tpr << std::endl;
  	std::cout << "tpr: " << tpr << std::endl;
	std::cout << "f1Score: " << f1Score << std::endl;

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );
    //imwrite( saveImageLocation, frame );

	return 0;
}

// recursive function which checks for the largest IOU and then checks the other ground truths
// to ensure two detected rectangles don't share the same ground truth.
void findCorrectIOU(int i, int numberOfFaces) {
	float chosenFaceIOU = 0;
	int indexOfChosenFace = 0; //corresponds to chosen face

	// chooses the face with the largest IOU
	for (int j = 0; j < numberOfFaces; j++) {
		if (ious[i][j] > chosenFaceIOU) {
			chosenFaceIOU = ious[i][j];
			indexOfChosenFace = j;
		}
	}

	// check for collisions
	if (chosenFaceIOU == 0) { // no corresponding ground truth was found
		GT_IOU_values[i] = 0;
	}
	else if (chosenFaces.count(indexOfChosenFace) == 0) { // no other ground truth uses this detected rectangle
		chosenFaces.insert({indexOfChosenFace, i});
		GT_IOU_values[i] = chosenFaceIOU;
	}
	else { // there was a collision (another ground truth uses this rectangle)
		int previousGroundTruth = chosenFaces[indexOfChosenFace];
		if (GT_IOU_values[previousGroundTruth] < chosenFaceIOU) {
			ious[previousGroundTruth][indexOfChosenFace] = 0;
			chosenFaces[indexOfChosenFace] = i;

			findCorrectIOU(previousGroundTruth, numberOfFaces);
			GT_IOU_values[i] = chosenFaceIOU;
		}
		else {
			ious[i][indexOfChosenFace] = 0;
			findCorrectIOU(i, numberOfFaces);
		}
	}
}

// Calculates the intersection over union of the two rectangles
float calculateIOU(Rect detectedRectangle, Rect groundTruthRectangle) {
	float x_overlap = max(0, min(detectedRectangle.x + detectedRectangle.width, groundTruthRectangle.x + groundTruthRectangle.width) - max(detectedRectangle.x, groundTruthRectangle.x));
	float y_overlap = max(0, min(detectedRectangle.y + detectedRectangle.height, groundTruthRectangle.y + groundTruthRectangle.height) - max(detectedRectangle.y, groundTruthRectangle.y));

	float intersection = x_overlap * y_overlap;
	float thisUnion = (detectedRectangle.width * detectedRectangle.height) + (groundTruthRectangle.width * groundTruthRectangle.height) - intersection;

	return (intersection / thisUnion);
}

// Cull the younglings
float calculateTpr() {
	int truePositives = 0;
	for (int i = 0; i < lengthGT; i++) {
		if (GT_IOU_values[i] > thresholdForCalculations) truePositives++;
	}
	return truePositives/lengthGT;
}

float calculateF1Score(int numberOfFaces) {
	int truePositives = 0;
	for (int i = 0; i < lengthGT; i++) {
		if (GT_IOU_values[i] > thresholdForCalculations) truePositives++;
	}

	int falseNegatives = lengthGT - truePositives;
	float precision = 0; 

 	if (numberOfFaces > 0) precision = (float) truePositives / (float) numberOfFaces;
	float recall = (float) truePositives / ((float) truePositives + (float) falseNegatives);
	float f1Score = 0;

	// if statement prevents division
	if (precision > 0) f1Score = (2 * ((precision * recall) / (precision + recall)) );

	return f1Score;
}

/** @function detectAndDisplay */
int detectAndDisplay( Mat frame ) {
	std::vector<Rect> faces;
	Mat canny_img;

	getCanny(frame, canny_img, 50, 100);
	violaJones(frame, faces);

	for (int i = 0; i < faces.size(); i++) {
		hough_circles(frame, canny_img, 20, 10, 300, faces[i]);
		hough_lines(frame, canny_img, 200, faces[i]);
	}
	

	

  	// Print number of faces found
	std::cout << faces.size() << std::endl;

	// populate ious table
	for (int i = 0; i < lengthGT; i++) {
		for (int j = 0; j < faces.size(); j++) {
			ious[i][j] = calculateIOU(faces[j], GTArray[i]);
		}
	}

    // Draw box around faces found
	for( int i = 0; i < faces.size(); i++ ) {
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

    // Draws the ground truth rectangles in red
	for ( int i = 0; i < lengthGT; i++) {
		rectangle(frame, Point(GTArray[i].x, GTArray[i].y),
						 Point(GTArray[i].x + GTArray[i].width,
						 GTArray[i].y + GTArray[i].height), Scalar( 0, 0, 255 ), 2);
	}

	return faces.size();
}
 
void violaJones( Mat frame, vector<Rect> &faces ) {
	Mat frame_gray;

	// Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.6, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
}

