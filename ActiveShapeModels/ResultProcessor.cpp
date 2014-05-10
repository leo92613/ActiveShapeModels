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
		cv::Point pt1(shapeY.at<double>(i), shapeX.at<double>(i));
		cv::Point pt2(shapeY.at<double>(next), shapeX.at<double>(next));

		cv::circle(image, pt1, 2, cv::Scalar(0, 69, 255), -1);
		cv::line(image, pt1, pt2, cv::Scalar(255, 191, 0)); 
	}

	cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);

	cv::imshow(windowName, image);
	cv::waitKey(0);
}

void ResultProcessor::debugLoadImage(const cv::Mat &originImage){
	originImage.copyTo(image);
	image.convertTo(image, CV_8U);
	cv::cvtColor(image, image, CV_GRAY2BGR);
}

void ResultProcessor::debugDrawLineOnImage(const cv::Point &_p1, const cv::Point &_p2){
	cv::Point p1(_p1.y, _p1.x), p2(_p2.y, _p2.x);
	cv::circle(image, p1, 2, cv::Scalar(0, 69, 255), -1);
	cv::circle(image, p2, 2, cv::Scalar(0, 69, 255), -1);
	cv::line(image, p1, p2, cv::Scalar(255, 191, 0));
	cv::namedWindow("debug", CV_WINDOW_AUTOSIZE);
	cv::imshow("debug", image);
}

void ResultProcessor::debugDrawCircleOnImage(const cv::Point &_pt, const double r, const int thickness){
	cv::Point pt(_pt.y, _pt.x);
	cv::circle(image, pt, r, cv::Scalar(230, 34, 55), thickness);
	cv::namedWindow("debug", CV_WINDOW_AUTOSIZE);
	cv::imshow("debug", image);
}

void ResultProcessor::debugDrawShapesOnImage(const cv::Mat &shapeX, const cv::Mat &shapeY){
	const int numberOfPoints = shapeX.rows;

	for(int i = 0; i < numberOfPoints; i++){
		int next = (i + 1) % numberOfPoints;
		cv::Point pt1(shapeY.at<double>(i), shapeX.at<double>(i));
		cv::Point pt2(shapeY.at<double>(next), shapeX.at<double>(next));

		cv::circle(image, pt1, 2, cv::Scalar(0, 69, 255), -1);
		cv::line(image, pt1, pt2, cv::Scalar(255, 191, 0)); 
	}

	cv::namedWindow("debug", CV_WINDOW_AUTOSIZE);

	cv::imshow("debug", image);
	cv::waitKey(0);
}