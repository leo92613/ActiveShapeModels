#pragma once

#include <opencv2\core\core.hpp>
#include <vector>
#include <opencv2\imgproc\imgproc.hpp>
#include "LocalFeature.h"
#include "AlignShape.h"
#include "TrainingData.h"
#include "Constant.h"
#include "PCAShapeModel.h"

class TrainingData
{
public:
	std::vector<cv::Mat> trainingImages, gradientImages;
	cv::Mat trainingShapesX, trainingShapesY;
	cv::Mat alignedShapesX, alignedShapesY;
	cv::Mat meanAlignedShapesX, meanAlignedShapesY;
	cv::Mat W, WInOneColumn;
	std::vector<LocalFeature> localFeatures;
	PCAShapeModel pcaShapeModel;

	TrainingData(void);
	~TrainingData(void);

	double getWk(int k);
	void generateWAndWInOneColumn();
	void generateGradientImages();
	void generateLocalFeatures();
	void generatePCAShapeModel();
	void alignShapes();

	void findBestShifts(const cv::Mat &shapeX, const cv::Mat &shapeY, const cv::Mat &gradientImage,
						cv::Mat &shiftsX, cv::Mat &shiftsY);
};

