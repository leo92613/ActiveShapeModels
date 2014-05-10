#pragma once

#include <opencv2\opencv.hpp>
#include <iostream>
using std::string;

class ResultProcessor
{
public:
	ResultProcessor(void);
	~ResultProcessor(void);

	void showResultImage(const cv::Mat &shapeX, const cv::Mat &shapeY, 
		const cv::Mat &originImage, const string &windowName);

	cv::Mat image;
	void debugLoadImage(const cv::Mat &originImage);
	void debugDrawLineOnImage(const cv::Point &p1, const cv::Point &p2);
	void debugDrawCircleOnImage(const cv::Point &pt, const double r, const int thickness = 1);
	void debugDrawShapesOnImage(const cv::Mat &shapeX, const cv::Mat &shapeY);
};

