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
void findCorrectIOU(int i, int numberOfboards);
float calculateTpr();
float calculateF1Score(int numberOfboards);
void violaJones(Mat frame, vector<Rect> &boards);

// Ground truth boards array -------------------------------------------------------------------------
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
int lengthGT = sizeof(dart1_ground)/sizeof(dart1_ground[0]);
Rect GTArray[sizeof(dart1_ground)/sizeof(dart1_ground[0])] = dart1_ground;
const float thresholdForCalculations = 0.45;

std::unordered_map<int, int> chosenboards; // index of chosen face, index of GT which chose face
float GT_IOU_values[sizeof(GTArray)/sizeof(GTArray[0])]; // array to store the correct IOUs for each GT
float ious[20][20];

/** @function main */
int main( int argc, const char** argv ) {
  	// 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect boards and Display Result
	int numberOfboards = detectAndDisplay( frame );

	// make sure each GT has only 1 associated face
	for (int i = 0; i < lengthGT; i++) {
		findCorrectIOU(i, numberOfboards);
	}

	// find TPR
	float tpr = calculateTpr();
	float f1Score = calculateF1Score(numberOfboards);
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
void findCorrectIOU(int i, int numberOfboards) {
	float chosenFaceIOU = 0;
	int indexOfChosenFace = 0; //corresponds to chosen face

	// chooses the face with the largest IOU
	for (int j = 0; j < numberOfboards; j++) {
		if (ious[i][j] > chosenFaceIOU) {
			chosenFaceIOU = ious[i][j];
			indexOfChosenFace = j;
		}
	}

	// check for collisions
	if (chosenFaceIOU == 0) { // no corresponding ground truth was found
		GT_IOU_values[i] = 0;
	}
	else if (chosenboards.count(indexOfChosenFace) == 0) { // no other ground truth uses this detected rectangle
		chosenboards.insert({indexOfChosenFace, i});
		GT_IOU_values[i] = chosenFaceIOU;
	}
	else { // there was a collision (another ground truth uses this rectangle)
		int previousGroundTruth = chosenboards[indexOfChosenFace];
		if (GT_IOU_values[previousGroundTruth] < chosenFaceIOU) {
			ious[previousGroundTruth][indexOfChosenFace] = 0;
			chosenboards[indexOfChosenFace] = i;

			findCorrectIOU(previousGroundTruth, numberOfboards);
			GT_IOU_values[i] = chosenFaceIOU;
		}
		else {
			ious[i][indexOfChosenFace] = 0;
			findCorrectIOU(i, numberOfboards);
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

float calculateF1Score(int numberOfboards) {
	int truePositives = 0;
	for (int i = 0; i < lengthGT; i++) {
		if (GT_IOU_values[i] > thresholdForCalculations) truePositives++;
	}

	int falseNegatives = lengthGT - truePositives;
	float precision = 0; 

 	if (numberOfboards > 0) precision = (float) truePositives / (float) numberOfboards;
	float recall = (float) truePositives / ((float) truePositives + (float) falseNegatives);
	float f1Score = 0;

	// if statement prevents division
	if (precision > 0) f1Score = (2 * ((precision * recall) / (precision + recall)) );

	return f1Score;
}


/** @function detectAndDisplay */
int detectAndDisplay( Mat frame ) {
	vector<Rect> violaBoards; // Contains the locations of rectangles detected by Viola Jones
	vector<Point3d> circle_centres; // Contains circles found by hough circle 
	vector<Rect> rejected_viola_boards; // Violas with no overlapping circles
	vector<Rect> rejected_circle_boards; // Circles with no overlapping violas

	vector<Rect> detectedBoards; // Final output boards

	Mat canny_img, directionImg, magnitudeImg;
	Rect imageDimensions = Rect(0, 0, frame.cols, frame.rows);

	Mat lines_image = frame.clone(); // Lines will be drawn to this image
	Mat circles_image = frame.clone(); // Surcools will be drawn to this image
	
	getCanny(frame, canny_img, 50, 100);
	getGradient(frame, directionImg, magnitudeImg);

	// Variables for detections
	int circleHoughThreshold = imageDimensions.area() / 37000;
	int maxRadius = max(imageDimensions.height, imageDimensions.width) * 0.63;
	int minRadius = maxRadius * 0.05;

	int circle_hough_space_dim[3] = {frame.rows + (2 * maxRadius), frame.cols + (2 * maxRadius), maxRadius - minRadius + 1}; // adding border padding of maxRadius
	Mat circle_hough_space(3, circle_hough_space_dim, CV_32F, Scalar(0));
	Mat flattened_circle_hough((frame.rows + (2 * maxRadius)), frame.cols + (2 * maxRadius), CV_32FC1);

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	std::cout << "OK im going to start! Buckle up buckeroo!" << endl; 

	// Creates boards from Viola Jones
	violaJones(frame, violaBoards);

	// Creates boards from cicles
	create_circle_houghspace(frame, canny_img, circle_hough_space, flattened_circle_hough, directionImg, minRadius, maxRadius, imageDimensions);
	draw_circles(circles_image, circle_hough_space, flattened_circle_hough, circle_centres, circleHoughThreshold, minRadius, maxRadius); // Draws and counts circles

	// cluster-fuck surcools
	

	// cluster-fuck violas

	
	// ------------------------------Check for overlap between circle and Viola boards -------------------------------------------------------------------
	for (int i = 0; i < violaBoards.size(); i++) {
		Rect best_iou_circle_rect;
		float best_iou = 0;

		for (int j = 0; j < circle_centres.size(); j++) {
			int r = circle_centres[j].z; // r = true radius of circle
			Rect circle_rect = Rect(circle_centres[j].x - r, circle_centres[j].y - r, 2*r, 2*r); // for each circle create a rect around it

			if (circle_centres[j].x >= violaBoards[i].x && circle_centres[j].x <= (violaBoards[i].x + violaBoards[i].width)
					&& circle_centres[j].y >= violaBoards[i].y && circle_centres[j].y <= (violaBoards[i].y + violaBoards[i].height)) { // if circle centre lies inside of viola detection
					
				float iou = calculateIOU(circle_rect, violaBoards[i]);

				if (iou < 0.8) { // every board with low iou given more processing
					rejected_circle_boards.push_back(circle_rect);
				}
				else {
					if (iou > best_iou) { // only one circle with iou greater than threshold is rejected
						best_iou_circle_rect = circle_rect;
						best_iou = iou;
					}
				}
			}
			else { // Circle does not intercept viola at all
				rejected_circle_boards.push_back(circle_rect);
			}	

		}
		if (best_iou_circle_rect.width != 0 || best_iou_circle_rect.height != 0) detectedBoards.push_back(best_iou_circle_rect); // if there was an overlapping iou add to detectedBoards
		else {
			rejected_viola_boards.push_back(violaBoards[i]); // else add viola to rejected violas 
		}
	}
	// -----------------------------------------------------------------------------------------------------------------------------------------------


	// process rejected violas
	for (int i = 0; i < rejected_viola_boards.size(); i++) { // if board has enough lines, add to detectedBoards
		int hough_line_threshold = rejected_viola_boards[i].area() / 600;
		//Rect rect = Rect(circle_centres[i].x - circle_centres[i].z, circle_centres[i].y - circle_centres[i].z, 2*circle_centres[i].z, 2*circle_centres[i].z);
		int line_count = hough_lines(canny_img, rejected_viola_boards[i], directionImg, lines_image, hough_line_threshold);
		if (line_count > 10) detectedBoards.push_back(rejected_viola_boards[i]);
	}

	// process rejected circles
		// # of lines in circle check




	// finally, combine overlapping detected boards

		
	cout << "There were: " << detectedBoards.size() << " refinedBoards!" << endl;
	
	// ---------------------Filters viola boards by circleCount---------------------
	// 	cout << "By the way, the circleCount for this face is: " << circleCount << endl;

	// 	if (circleCount > 5) {
	// 		int lineHoughThreshold = boards[i].area()/500; 

	// 		cout << "houghThreshold = " << lineHoughThreshold << endl;
			
	// 		hough_lines(frame, canny_img, lines_image, lineHoughThreshold, boards[i]);
	// 	}
	// 	cout << (i+1) << " mississipi, " << endl;
	// }
	// ------------------------------------------------------------------------------

	imwrite("hough_lines.jpg", lines_image);
	imwrite("hough_circles.jpg", circles_image);
	cout << "Wow that was fun! See you later aha ;)" << endl;


	// -----------------------------------Scorings-----------------------------------

	// populate ious table
	for (int i = 0; i < lengthGT; i++) {
		for (int j = 0; j < violaBoards.size(); j++) {
			ious[i][j] = calculateIOU(violaBoards[j], GTArray[i]);
		}
	}
    // Draw box around boards found
	for( int i = 0; i < violaBoards.size(); i++ ) {
		rectangle(frame, Point(violaBoards[i].x, violaBoards[i].y), Point(violaBoards[i].x + violaBoards[i].width, violaBoards[i].y + violaBoards[i].height), Scalar( 0, 255, 0 ), 2);
	}
	// // Draw box around boards found
	// for( int i = 0; i < refinedBoardsUsingLines.size(); i++ ) {
	// 	rectangle(frame, Point(refinedBoardsUsingLines[i].x, refinedBoardsUsingLines[i].y), Point(refinedBoardsUsingLines[i].x + refinedBoardsUsingLines[i].width, refinedBoardsUsingLines[i].y + refinedBoardsUsingLines[i].height), Scalar( 255, 255, 255 ), 2);
	// }

    // Draws the ground truth rectangles in red
	for ( int i = 0; i < lengthGT; i++) {
		rectangle(frame, Point(GTArray[i].x, GTArray[i].y),
						 Point(GTArray[i].x + GTArray[i].width,
						 GTArray[i].y + GTArray[i].height), Scalar( 0, 0, 255 ), 2);
	}
	// ------------------------------------------------------------------------------
	return violaBoards.size();
}
 
void violaJones( Mat frame, vector<Rect> &boards ) {
	Mat frame_gray;

	// Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, boards, 1.6, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
}

