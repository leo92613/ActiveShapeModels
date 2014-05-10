#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <vector>
#include "Constant.h"

class LocalFeature
{
public:
	LocalFeature(void);
	~LocalFeature(void);
	
	cv::Mat covar, mean, icovar;

	void computeLocalFeature(const cv::Mat &trainingShapesX, const cv::Mat &trainingShapesY,
		const std::vector<cv::Mat> &gradientImages, const int curr);
	void findBestShift(const cv::Mat &shapeX, const cv::Mat &shapeY, const cv::Mat &gradientImage,
				   const int curr, double &shiftX, double &shiftY);

	void caculateDxDy(const double x1, const double y1,
		const double x2, const double y2, double &dx, double &dy);
};

