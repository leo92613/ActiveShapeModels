#include "ResultProcessor.h"
//debug
#include <iostream>
using std::cout;
using std::endl;
//end debug

ResultProcessor::ResultProcessor(void)
{
}


ResultProcessor::~ResultProcessor(void)
{
}

void ResultProcessor::showResultImage(const cv::Mat &shapeX, const cv::Mat &shapeY, const cv::Mat &originImage, const string &windowName){
	const int numberOfPoints = shapeX.rows;

	cv::Mat image;
	originImage.convertTo(image, CV_8UC1);
	cv::cvtColor(image, image, CV_GRAY2BGR);

	for(int i = 0; i < numberOfPoints; i++){
		int next = (i + 1) % numberOfPoints;
		cv::Point pt1(shapeX.at<double>(i), shapeY.at<double>(i));
		cv::Point pt2(shapeX.at<double>(next), shapeY.at<double>(next));

		cv::circle(image, pt1, 2, cv::Scalar(0, 69, 255), -1);
		cv::line(image, pt1, pt2, cv::Scalar(255, 191, 0)); 
	}

	cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	//debug
	//cout << "image : " << endl;
	//cout << image << endl;
	//end debug
	cv::imshow(windowName, image);
	cv::waitKey(0);
}