#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include <iostream>
#include <vector> 
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

using namespace std;
cv::Ptr<cv::ml::KNearest> KN = cv::ml::KNearest::create();

// class possibleplate
class Possible_Plate {
public:
	cv::Mat plate;
	cv::Mat plate_gray;
	cv::Mat thresh;
	cv::RotatedRect relocation;
	std::string str_c;

	static bool sort_des_order(const Possible_Plate &ppLeft, const Possible_Plate &ppRight) {
		return(ppLeft.str_c.length() > ppRight.str_c.length());
	}
};
// class possible char
class possiblechar {
public:
	std::vector<cv::Point> contour;
	cv::Rect rect;
	int centroid_x;
	int centroid_y;
	double diag;
	double ratio;

	static bool sort(const possiblechar &pcLeft, const possiblechar & pcRight) {
		return(pcLeft.centroid_x < pcRight.centroid_x);
	}

	bool operator == (const possiblechar& others) const {
		if (this->contour == others.contour) return true;
		else return false;
	}

	bool operator != (const possiblechar& others) const {
		if (this->contour != others.contour) return true;
		else return false;
	}
	// init
	possiblechar(std::vector<cv::Point> _contour) {
		contour = _contour;
		rect = cv::boundingRect(contour);
		centroid_x = (rect.x + rect.x + rect.width) / 2;
		centroid_y = (rect.y + rect.y + rect.height) / 2;
		diag = sqrt(pow(rect.width, 2) + pow(rect.height, 2));
		ratio = (float)rect.width / (float)rect.height;
	}
};

// functions
// pre-process
void preprocess(cv::Mat &original, cv::Mat &gray, cv::Mat &thresh);
cv::Mat extractValue(cv::Mat &imgOriginal);
cv::Mat maximizeContrast(cv::Mat &imgGrayscale);
// detect char
bool loadK_TrainK();
std::vector<Possible_Plate> detectCharsInPlates(std::vector<Possible_Plate> &vectorOfPossiblePlates);
std::vector<possiblechar> findPossibleCharsInPlate(cv::Mat &imgGrayscale, cv::Mat &imgThresh);
std::vector<possiblechar> removeInnerOverlappingChars(std::vector<possiblechar> &vectorOfMatchingChars);
std::string recognizeCharsInPlate(cv::Mat &imgThresh, std::vector<possiblechar> &vectorOfMatchingChars);
std::vector<std::vector<possiblechar> > MatchedCharDetecting(const std::vector<possiblechar> &vector_possibleChars);
std::vector<possiblechar> findVectorOfMatchingChars(const possiblechar &possibleChar, const std::vector<possiblechar> &vectorOfChars);
double distanceBetweenChars(const possiblechar &firstChar, const possiblechar &secondChar);
double angleBetweenChars(const possiblechar &firstChar, const possiblechar &secondChar);
bool checkIfPossibleChar(possiblechar &possibleChar);

// detect plate
std::vector<Possible_Plate> platesDetecting(cv::Mat &original);
std::vector<possiblechar> possiblecharDetecting(cv::Mat &thresh);
Possible_Plate extract_plate(cv::Mat &original, std::vector<possiblechar> &matched_chars);



void preprocess(cv::Mat &original, cv::Mat &gray, cv::Mat &thresh) {
	gray = extractValue(original);                           // get imgGrayscale
	cv::Mat imgMaxContrastGrayscale = maximizeContrast(gray);       // maximize contrast with top hat and black hat
	cv::Mat imgBlurred;
	cv::GaussianBlur(imgMaxContrastGrayscale, imgBlurred, cv::Size(5, 5), 0);          // gaussian blur
																									// imgThresh
	cv::adaptiveThreshold(imgBlurred, thresh, 255.0, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 19, 9);
}

cv::Mat maximizeContrast(cv::Mat &imgGrayscale) {
	cv::Mat imgTopHat;
	cv::Mat imgBlackHat;
	cv::Mat imgGrayscalePlusTopHat;
	cv::Mat imgGrayscalePlusTopHatMinusBlackHat;
	cv::Mat structuringElement = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3));
	cv::morphologyEx(imgGrayscale, imgTopHat, CV_MOP_TOPHAT, structuringElement);
	cv::morphologyEx(imgGrayscale, imgBlackHat, CV_MOP_BLACKHAT, structuringElement);
	imgGrayscalePlusTopHat = imgGrayscale + imgTopHat;
	imgGrayscalePlusTopHatMinusBlackHat = imgGrayscalePlusTopHat - imgBlackHat;
	return imgGrayscalePlusTopHatMinusBlackHat;
}

cv::Mat extractValue(cv::Mat &imgOriginal) {
	cv::Mat imgHSV;
	std::vector<cv::Mat> vectorOfHSVImages;
	cv::Mat imgValue;
	cv::cvtColor(imgOriginal, imgHSV, CV_BGR2HSV);
	cv::split(imgHSV, vectorOfHSVImages);
	imgValue = vectorOfHSVImages[2];
	return imgValue;
}

bool loadK_TrainK() {
	// read in training classifications
	cv::Mat classificationInts;
	// open the file
	cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);
	if (fsClassifications.isOpened() == false) {
		std::cout << "error, unable to open training classifications file, exiting program\n\n";
		return(false);
	}
	fsClassifications["classifications"] >> classificationInts;
	fsClassifications.release();

	cv::Mat mat_TrainingImages;
	// open training image file
	cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);
	if (fsTrainingImages.isOpened() == false) {
		std::cout << "failed to open file";
		return false;
	}
	fsTrainingImages["images"] >> mat_TrainingImages;
	fsTrainingImages.release();

	KN->setDefaultK(1);
	// training
	KN->train(mat_TrainingImages, cv::ml::ROW_SAMPLE, classificationInts);
	return true;
}

std::vector<Possible_Plate> detectCharsInPlates(std::vector<Possible_Plate> &vectorOfPossiblePlates) {
	int counter = 0;
	cv::Mat imgContours;
	std::vector<std::vector<cv::Point>> contours;
	cv::RNG rng;
	if (vectorOfPossiblePlates.empty()) {               
		return(vectorOfPossiblePlates);               
	}
	for (auto &possibleP : vectorOfPossiblePlates) {
		preprocess(possibleP.plate, possibleP.plate_gray, possibleP.thresh);

		cv::resize(possibleP.thresh, possibleP.thresh, cv::Size(), 1.6, 1.6);
		cv::threshold(possibleP.thresh, possibleP.thresh, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);

		// find all possible chars in the plate, first finds all contours, then only includes contours that could be chars 
		std::vector<possiblechar> vectorOfPossibleCharsInPlate = findPossibleCharsInPlate(possibleP.plate_gray, possibleP.thresh);
		imgContours = cv::Mat(possibleP.thresh.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
		contours.clear();
		for (auto &possibleChar : vectorOfPossibleCharsInPlate) {
			contours.push_back(possibleChar.contour);
		}
		cv::drawContours(imgContours, contours, -1, cv::Scalar(255.0, 255.0, 255.0));

		std::vector<std::vector<possiblechar> > vectorOfVectorsOfMatchingCharsInPlate = MatchedCharDetecting(vectorOfPossibleCharsInPlate);
		imgContours = cv::Mat(possibleP.thresh.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));

		contours.clear();

		for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) {
			int intRandomBlue = rng.uniform(0, 256);
			int intRandomGreen = rng.uniform(0, 256);
			int intRandomRed = rng.uniform(0, 256);

			for (auto &matchingChar : vectorOfMatchingChars) {
				contours.push_back(matchingChar.contour);
			}
			cv::drawContours(imgContours, contours, -1, cv::Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));
		}

		if (vectorOfVectorsOfMatchingCharsInPlate.size() == 0) {
			possibleP.str_c = "";
			continue;
		}
		for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) {              
			// for each vector of matching chars in the current plate
			std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), possiblechar::sort);     
			// sort the chars left to right
			vectorOfMatchingChars = removeInnerOverlappingChars(vectorOfMatchingChars);                                     
			// eliminate  overlapping chars
		}
		imgContours = cv::Mat(possibleP.thresh.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));

		for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) {
			int intRandomBlue = rng.uniform(0, 256);
			int intRandomGreen = rng.uniform(0, 256);
			int intRandomRed = rng.uniform(0, 256);

			contours.clear();

			for (auto &matchingChar : vectorOfMatchingChars) {
				contours.push_back(matchingChar.contour);
			}
			cv::drawContours(imgContours, contours, -1, cv::Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));
		}

		//  the longest vector should be our plate
		unsigned int intLenOfLongestVectorOfChars = 0;
		unsigned int intIndexOfLongestVectorOfChars = 0;
		// loop to get the longest
		for (unsigned int i = 0; i < vectorOfVectorsOfMatchingCharsInPlate.size(); i++) {
			if (vectorOfVectorsOfMatchingCharsInPlate[i].size() > intLenOfLongestVectorOfChars) {
				intLenOfLongestVectorOfChars = vectorOfVectorsOfMatchingCharsInPlate[i].size();
				intIndexOfLongestVectorOfChars = i;
			}
		}
		std::vector<possiblechar> longestVectorOfMatchingCharsInPlate = vectorOfVectorsOfMatchingCharsInPlate[intIndexOfLongestVectorOfChars];
		imgContours = cv::Mat(possibleP.thresh.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));

		contours.clear();

		for (auto &matchingChar : longestVectorOfMatchingCharsInPlate) {
			contours.push_back(matchingChar.contour);
		}
		cv::drawContours(imgContours, contours, -1, cv::Scalar(255.0, 255.0, 255.0));

		possibleP.str_c = recognizeCharsInPlate(possibleP.thresh, longestVectorOfMatchingCharsInPlate);
		counter++;
	}
	return vectorOfPossiblePlates;
}

std::vector<possiblechar> findPossibleCharsInPlate(cv::Mat &imgGrayscale, cv::Mat &imgThresh) {
	std::vector<possiblechar> vectorOfPossibleChars;                           
	cv::Mat imgThreshCopy;
	std::vector<std::vector<cv::Point> > contours;
	imgThreshCopy = imgThresh.clone();				
	cv::findContours(imgThreshCopy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);        // find all contours in plate
	for (auto &contour : contours) {                            
		possiblechar possibleChar(contour);

		if (checkIfPossibleChar(possibleChar)) {               
			vectorOfPossibleChars.push_back(possibleChar);      // add to vector of possible chars
		}
	}
	return vectorOfPossibleChars;
}

std::vector<possiblechar> removeInnerOverlappingChars(std::vector<possiblechar> &vectorOfMatchingChars) {
	std::vector<possiblechar> vectorOfMatchingCharsWithInnerCharRemoved(vectorOfMatchingChars);
	for (auto &currentChar : vectorOfMatchingChars) {
		for (auto &otherChar : vectorOfMatchingChars) {
			if (currentChar != otherChar) {                         

				if (distanceBetweenChars(currentChar, otherChar) < (currentChar.diag * 0.3)) {

					// if current char is smaller than other char
					if (currentChar.rect.area() < otherChar.rect.area()) {
						// look for char in vector with an iterator
						std::vector<possiblechar>::iterator currentCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), currentChar);
						if (currentCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
							vectorOfMatchingCharsWithInnerCharRemoved.erase(currentCharIterator);       
						}
					}
					else {        
						std::vector<possiblechar>::iterator otherCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), otherChar);
						// if iterator did not get to end, then the char was found in the vector
						if (otherCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
							vectorOfMatchingCharsWithInnerCharRemoved.erase(otherCharIterator);         // so remove the char
						}
					}
				}
			}
		}
	}
	return vectorOfMatchingCharsWithInnerCharRemoved;
}

std::string recognizeCharsInPlate(cv::Mat &imgThresh, std::vector<possiblechar> &vectorOfMatchingChars) {
	std::string res;
	cv::Mat thresh_color;
	std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), possiblechar::sort);
	cv::cvtColor(imgThresh, thresh_color, CV_GRAY2BGR);
	for (auto &currentChar : vectorOfMatchingChars) {           
		cv::rectangle(thresh_color, currentChar.rect, cv::Scalar(0.0, 255.0, 0.0), 2);       // draw green box around the char
		cv::Mat imgROItoBeCloned = imgThresh(currentChar.rect);                 // get ROI image of bounding rect
		cv::Mat imgROI = imgROItoBeCloned.clone();      
		cv::Mat imgROIResized;
		// resize image, this is necessary for char recognition
		cv::resize(imgROI, imgROIResized, cv::Size(20, 30));
		cv::Mat matROIFloat;
		imgROIResized.convertTo(matROIFloat, CV_32FC1);         
		cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);       // flatten Matrix into one row
		cv::Mat matCurrentChar(0, 0, CV_32F);                   
		KN->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     // finally we can call find_nearest !!!
		float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);       
		res = res + char(int(fltCurrentChar));        // append current char to full string
	}
	return res;
}

std::vector<std::vector<possiblechar> > MatchedCharDetecting(const std::vector<possiblechar> &vector_possibleChars) {
	std::vector<std::vector<possiblechar> > vector_of_vectors_chars;
	for (auto &pc : vector_possibleChars) {
		std::vector<possiblechar> vectors_chars = findVectorOfMatchingChars(pc, vector_possibleChars);
		vectors_chars.push_back(pc);
		if (vectors_chars.size() < 3)
			continue;
		vector_of_vectors_chars.push_back(vectors_chars);
		std::vector<possiblechar> vectorOfPossibleCharsWithCurrentMatchesRemoved;
		for (auto &possChar : vector_possibleChars) {
			if (std::find(vectors_chars.begin(), vectors_chars.end(), possChar) == vectors_chars.end()) {
				vectorOfPossibleCharsWithCurrentMatchesRemoved.push_back(possChar);
			}
		}
		// declare new vector of vectors of chars to get result from recursive call
		std::vector<std::vector<possiblechar> > recursiveVectorOfVectorsOfMatchingChars;
		recursiveVectorOfVectorsOfMatchingChars = MatchedCharDetecting(vectorOfPossibleCharsWithCurrentMatchesRemoved);
		for (auto &recursiveVectorOfMatchingChars : recursiveVectorOfVectorsOfMatchingChars) {      
			vector_of_vectors_chars.push_back(recursiveVectorOfMatchingChars);            
		}
		break;
	}
	return vector_of_vectors_chars;
}

std::vector<possiblechar> findVectorOfMatchingChars(const possiblechar &possibleChar, const std::vector<possiblechar> &vectorOfChars) {
	std::vector<possiblechar> vectorOfMatchingChars;              

	for (auto &possibleMatchingChar : vectorOfChars) {              

		if (possibleMatchingChar == possibleChar) {
			continue;           
		}
		// see if chars are a match
		double dblDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar);
		double dblAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar);
		double dblChangeInArea = (double)abs(possibleMatchingChar.rect.area() - possibleChar.rect.area()) / (double)possibleChar.rect.area();
		double dblChangeInWidth = (double)abs(possibleMatchingChar.rect.width - possibleChar.rect.width) / (double)possibleChar.rect.width;
		double dblChangeInHeight = (double)abs(possibleMatchingChar.rect.height - possibleChar.rect.height) / (double)possibleChar.rect.height;

		// check if chars match
		if (dblDistanceBetweenChars < (possibleChar.diag * 4.9) &&
			dblAngleBetweenChars < 11 &&
			dblChangeInArea < 0.5 &&
			dblChangeInWidth < 0.8 &&
			dblChangeInHeight < 0.2) {
			vectorOfMatchingChars.push_back(possibleMatchingChar);      // if the chars are a match, add the current char to vector of matching chars
		}
	}

	return(vectorOfMatchingChars);          // return result
}

double distanceBetweenChars(const possiblechar &firstChar, const possiblechar &secondChar) {
	int intX = abs(firstChar.centroid_x - secondChar.centroid_x);
	int intY = abs(firstChar.centroid_y - secondChar.centroid_y);
	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

double angleBetweenChars(const possiblechar &firstChar, const possiblechar &secondChar) {
	double dblAdj = abs(firstChar.centroid_x - secondChar.centroid_x);
	double dblOpp = abs(firstChar.centroid_y - secondChar.centroid_y);
	double dblAngleInRad = atan(dblOpp / dblAdj);
	double dblAngleInDeg = dblAngleInRad * (180.0 / CV_PI);
	return(dblAngleInDeg);
}

bool checkIfPossibleChar(possiblechar &possibleChar) {
	// rough check if it is possible 
	if (possibleChar.rect.area() > 80 &&
		possibleChar.rect.width > 2 && possibleChar.rect.height > 8 &&
		0.25 < possibleChar.ratio && possibleChar.ratio < 1.0) {
		return true;
	}
	else {
		return false;
	}
}

Possible_Plate extract_plate(cv::Mat &original, std::vector<possiblechar> &matched_chars) {
	Possible_Plate possibleps;
	std::sort(matched_chars.begin(), matched_chars.end(), possiblechar::sort);
	// get the centroid
	double c_x = (double)(matched_chars[0].centroid_x + matched_chars[matched_chars.size() - 1].centroid_x) / 2.0;
	double c_y = (double)(matched_chars[0].centroid_y + matched_chars[matched_chars.size() - 1].centroid_y) / 2.0;
	cv::Point2d p2dPlateCenter(c_x, c_y);
	// calculate the width and height
	int p_width = (int)((matched_chars[matched_chars.size() - 1].rect.x + matched_chars[matched_chars.size() - 1].rect.width - matched_chars[0].rect.x) * 1.3);
	double total_height = 0;
	for (auto &matchingChar : matched_chars) {
		total_height = total_height + matchingChar.rect.height;
	}
	double average_height = (double)total_height / matched_chars.size();
	int p_height = (int)(average_height * 1.5);
	// calculate angle of plate region
	double dblOpposite = matched_chars[matched_chars.size() - 1].centroid_y - matched_chars[0].centroid_y;
	double dblHypotenuse = distanceBetweenChars(matched_chars[0], matched_chars[matched_chars.size() - 1]);
	double dblCorrectionAngleInRad = asin(dblOpposite / dblHypotenuse);
	double dblCorrectionAngleInDeg = dblCorrectionAngleInRad * (180.0 / CV_PI);
	possibleps.relocation = cv::RotatedRect(p2dPlateCenter, cv::Size2f((float)p_width, (float)p_height), (float)dblCorrectionAngleInDeg);
	cv::Mat rotationM;
	cv::Mat rotated;
	cv::Mat cropped;
	rotationM = cv::getRotationMatrix2D(p2dPlateCenter, dblCorrectionAngleInDeg, 1.0);
	cv::warpAffine(original, rotated, rotationM, original.size());
	cv::getRectSubPix(rotated, possibleps.relocation.size, possibleps.relocation.center, cropped);
	possibleps.plate = cropped;
	return possibleps;
}

std::vector<Possible_Plate> platesDetecting(cv::Mat &original) {
	std::vector<Possible_Plate> plates;
	cv::Mat sceneGray;
	cv::Mat sceneThreshed;
	cv::Mat imgContours(original.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
	cv::RNG rng;
	cv::destroyAllWindows();

	cv::imshow("0", original);

		preprocess(original, sceneGray, sceneThreshed);        // preprocess to get grayscale and threshold images

	cv::imshow("1-a", sceneGray);
	cv::imshow("1-b", sceneThreshed);

	std::vector<possiblechar> vector_possiblechar = possiblecharDetecting(sceneThreshed);
	imgContours = cv::Mat(original.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
	std::vector<std::vector<cv::Point>> contours;
	for (auto &pc : vector_possiblechar) {
		contours.push_back(pc.contour);
	}

	cv::drawContours(imgContours, contours, -1, cv::Scalar(255.0, 255.0, 255.0));
	cv::imshow("2b", imgContours);

	std::vector<std::vector<possiblechar> > matching_chars = MatchedCharDetecting(vector_possiblechar);
	imgContours = cv::Mat(original.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
	for (auto &matching_char : matching_chars) {
		int intRandomBlue = rng.uniform(0, 256);
		int intRandomGreen = rng.uniform(0, 256);
		int intRandomRed = rng.uniform(0, 256);

		std::vector<std::vector<cv::Point> > contours;

		for (auto &m_char : matching_char) {
			contours.push_back(m_char.contour);
		}
		cv::drawContours(imgContours, contours, -1, cv::Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));
	}

	cv::imshow("3", imgContours);
	for (auto &matched_chars : matching_chars) {
		Possible_Plate possiblePlate = extract_plate(original, matched_chars);

		if (possiblePlate.plate.empty() == false) {
			plates.push_back(possiblePlate);
		}
	}
	cv::imshow("4-a", imgContours);

	for (unsigned int i = 0; i < plates.size(); i++) {
		cv::Point2f p2fRectPoints[4];
		plates[i].relocation.points(p2fRectPoints);
		for (int j = 0; j < 4; j++) {
			cv::line(imgContours, p2fRectPoints[j], p2fRectPoints[(j + 1) % 4], cv::Scalar(0.0, 0.0, 255.0), 2);
		}
		cv::imshow("4b", plates[i].plate);
		cv::waitKey(0);
	}
	return plates;
}

std::vector<possiblechar> possiblecharDetecting(cv::Mat &thresh) {
	std::vector<possiblechar> result;
	cv::Mat imgContours(thresh.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));
	int count = 0;
	cv::Mat thresh_copy = thresh.clone();
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(thresh_copy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	for (unsigned int i = 0; i < contours.size(); i++) {
		possiblechar possiblechar(contours[i]);
		if (checkIfPossibleChar(possiblechar)) {             
			count++;                         
			result.push_back(possiblechar);     
		}
	}

	return result;
}

int main(void) {
	bool blink_train_suc = loadK_TrainK();
	if (blink_train_suc == false) {
		return 0;
	}

	cv::Mat scene;
	scene = cv::imread("test2.png");
	if (scene.empty())
	{
		return 0;
	}

	//detect plates
	std::vector<Possible_Plate> vector_possible_plates = platesDetecting(scene);
	vector_possible_plates = detectCharsInPlates(vector_possible_plates);
	cv::imshow("original scene", scene);
	if (vector_possible_plates.empty()) {
	}
	else {
		std::sort(vector_possible_plates.begin(), vector_possible_plates.end(), Possible_Plate::sort_des_order);//sorting
		Possible_Plate p = vector_possible_plates.front();
		cv::imshow("img plate", p.plate);
		cv::imshow("img thresh", p.thresh);
		if (p.str_c.length() == 0) {
			return 0;
		}

		std::cout << std::endl << "license plate " << p.str_c << std::endl;
		cv::imwrite("scene.png", scene);
	}
	cv::waitKey(0);
	return 0;
}