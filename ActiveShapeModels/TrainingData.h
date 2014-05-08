#pragma once

#include <opencv2\core\core.hpp>
#include <vector>
#include <string>
#include <opencv2\imgproc\imgproc.hpp>
#include "LocalFeature.h"
#include "AlignShape.h"
#include "TrainingData.h"
#include "Constant.h"
#include "PCAShapeModel.h"
#include "FileManager.h"

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
	TrainingData(const string &csvFilename, const string &imagesDir);
	
	~TrainingData(void);

	void loadDataAndImagesFromCSV(const string &csvFilename, const string &imagesDir);

	double getWk(int k);
	void generateWAndWInOneColumn();
	void generateGradientImages();
	void generateLocalFeatures();
	void generateAlignedShapes();
	void generatePCAShapeModel();

	void findBestShifts(const cv::Mat &shapeX, const cv::Mat &shapeY, const cv::Mat &gradientImage,
						cv::Mat &shiftsX, cv::Mat &shiftsY);
	void findBestDeforming(const cv::Mat &X0, const cv::Mat &Y0, const cv::Mat &sX, const cv::Mat &sY,
		 const MappingParameters &_para, cv::Mat &resX, cv::Mat &resY);
};

