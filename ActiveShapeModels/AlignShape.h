#pragma once

#include <opencv2\core\core.hpp>
#include <vector>
#include "MappingParameters.h"

class AlignShape
{
public:
	AlignShape(void);
	~AlignShape(void);

	MappingParameters findBestMapping(const cv::Mat &Ax, const cv::Mat &Ay, 
		const cv::Mat &Bx, const cv::Mat By, const cv::Mat WInOneColumn, const cv::Mat W);

	void caculateNewCoordinatesForTrainingShapes(const cv::Mat &shapesX, const cv::Mat &shapesY, 
		std::vector<MappingParameters> &P, cv::Mat &newShapesX, cv::Mat &newShapesY);

	void getMeanShape(const cv::Mat &shapesX, const cv::Mat &shapesY, cv::Mat &meanShape);
};

