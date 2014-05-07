#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "TrainingData.h"
#include "Constant.h"
#include "AlignShape.h"

class ActiveShapeModels
{
public:
	cv::Mat image, gradiantImage;
	cv::Mat shapeX, shapeY;

	void creatInitialShape();
	void iterationSearch(TrainingData &trainingData);

	ActiveShapeModels(void);
	~ActiveShapeModels(void);
};

