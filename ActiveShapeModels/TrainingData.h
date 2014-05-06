#pragma once

#include <opencv2\core\core.hpp>
#include <vector>
#include <opencv2\imgproc\imgproc.hpp>


class TrainingData
{
public:
	std::vector<cv::Mat> trainingImages, gradientImages;
	cv::Mat trainingShapesX, trainingShapesY;
	cv::Mat alignedShapesX, alignedShapesY;
	cv::Mat meanAlignedShapesX, meanAlignedShapesY;
	cv::Mat W, WInOneColumn;

	TrainingData(void);
	~TrainingData(void);

	double getWk(int k);
	void generateWAndWInOneColumn();
	void generateGradientImages();
};

