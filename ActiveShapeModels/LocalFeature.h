#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

const int _featureSize = 5;

class LocalFeature
{
public:
	LocalFeature(void);
	~LocalFeature(void);
	
	cv::Mat covar, mean;

	void computeLocalFeature(const cv::Mat &trainingShapesX, const cv::Mat &trainingShapesY,
		const cv::Mat &gradientImages, const int curr);
	void findBestShift(const cv::Mat &shapeX, const cv::Mat &shapeY, const cv::Mat &gradImage, 
		const int curr, cv::Mat &shiftX, cv::Mat &shiftY);
};

