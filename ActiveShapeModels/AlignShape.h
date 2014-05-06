#pragma once

#include <opencv2\core\core.hpp>
#include <vector>
#include "MappingParameters.h"

class AlignShape
{
public:
	AlignShape(void);
	~AlignShape(void);

	MappingParameters AlignShape::findBestMapping(const cv::Mat &Ax, const cv::Mat &Ay, 
		const cv::Mat &Bx, const cv::Mat &By, const cv::Mat &WInOneColumn, const cv::Mat &W);

	void caculateNewCoordinatesForTrainingShapes(const cv::Mat &shapesX, const cv::Mat &shapesY, 
		std::vector<MappingParameters> &P, cv::Mat &newShapesX, cv::Mat &newShapesY);

	void getMeanShape(const cv::Mat &shapesX, const cv::Mat &shapesY, cv::Mat &meanShapeX, cv::Mat &meanShapeY);

	double getDistanceOfTwoShapes(const cv::Mat &A, const cv::Mat &B);

	void alignTrainingShapes(const cv::Mat &trainingShapesX, const cv::Mat &trainingShapesY, 
		const cv::Mat &WInOneColumn, const cv::Mat &W, const int iterationTimeThreshold, const double convergencyThreshold,
		cv::Mat &newShapesX, cv::Mat &newShapesY, cv::Mat &meanShapeX, cv::Mat &meanShapeY);
};

