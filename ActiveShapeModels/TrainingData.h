#pragma once

#include <opencv2\core\core.hpp>

class TrainingData
{
public:
	cv::Mat trainingShapesX, trainingShapesY;
	cv::Mat alignedShapesX, alignedShapesY;
	cv::Mat meanAlignedShapesX, meanAlignedShapesY;
	cv::Mat W, WInOneColumn;

	TrainingData(void);
	~TrainingData(void);

	double getWk(int k);
	void generateWAndWInOneColumn();
};

