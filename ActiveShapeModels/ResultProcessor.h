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
};

