//training 
#include<opencv2/core/core.hpp>
#include<iostream>
#include<vector>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>




int main() {

	cv::Mat input;         
	cv::Mat gray;                
	cv::Mat blur;                 
	cv::Mat binary;                  
	cv::Mat copy;              

	std::vector<std::vector<cv::Point> > contours;        //  contours vector
	std::vector<cv::Vec4i> Hierarchy;                    // contours hierarchy vector

	

	std::vector<int> possiblechar = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
		'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W', 'X', 'Y', 'Z' };

	input = cv::imread("tem1.jpg");   



	cv::cvtColor(input, gray, CV_BGR2GRAY);        // convert to grayscale
	cv::GaussianBlur(gray,blur,cv::Size(5, 5), 0);                          

	cv::adaptiveThreshold(blur,binary,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY_INV,11,2);                                    

	cv::imshow("binary", binary);         

	copy = binary.clone();

	cv::Mat classification;
	cv::Mat matfloat;

	cv::findContours(copy, contours, Hierarchy,cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);               
	for (int i = 0; i < contours.size(); i++) {                          
		if (cv::contourArea(contours[i]) > 80) {               
			cv::Rect rect = cv::boundingRect(contours[i]);              

			cv::Mat ROI = binary(rect);           
			cv::Mat matROIResized;
			cv::resize(ROI, matROIResized, cv::Size(20, 25));     
			cv::imshow("ROI", ROI);                               

			int intChar = cv::waitKey(0);           

			 if (std::find(possiblechar.begin(), possiblechar.end(), intChar) != possiblechar.end()) {     
				classification.push_back(intChar);      
				cv::Mat imgflot;                         
				matROIResized.convertTo(imgflot, CV_32FC1);
				cv::Mat matImageFlattenedFloat = imgflot.reshape(1, 1);
				matfloat.push_back(matImageFlattenedFloat);       
			}   
		} 
	}   


	// save classifications to file and image to files
	cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);           
	fsClassifications << "classifications" << classification;        
	fsClassifications.release();                                            
	cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);         
	fsTrainingImages << "images" << matfloat;         
	fsTrainingImages.release();                                                

	return(0);
}



