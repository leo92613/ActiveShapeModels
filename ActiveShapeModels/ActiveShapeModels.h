#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "TrainingData.h"
#include "Constant.h"
#include "AlignShape.h"
#include "FileManager.h"

class ActiveShapeModels
{
public:
	cv::Mat image, gradiantImage;
	cv::Mat shapeX, shapeY;

	void loadImage(const string &filename);
	void generateGradiantImage();
	void creatInitialShape(TrainingData &trainingData);
	void iterationSearch(TrainingData &trainingData);

	ActiveShapeModels(void);
	ActiveShapeModels(const string &filename);
	~ActiveShapeModels(void);
};

