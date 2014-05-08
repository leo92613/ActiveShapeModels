#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "Constant.h"
#include "MappingParameters.h"
#include "AlignShape.h"

class PCAShapeModel
{
public:
	PCAShapeModel(void);
	~PCAShapeModel(void);

	cv::PCA pca;
	cv::Mat meanShapeX, meanShapeY;

	void generateBases(const cv::Mat &alignedShapesX, const cv::Mat &alignedShapesY, 
		const cv::Mat &meanShapeX, const cv::Mat &meanShapeY);
	
	void mergeXY(const cv::Mat &X, const cv::Mat &Y, cv::Mat &XY);

	void splitXY(const cv::Mat &XY, cv::Mat &X, cv::Mat &Y);
	
	void findBestDeforming(const cv::Mat &X0, const cv::Mat &Y0,
		const cv::Mat &sX, const cv::Mat &sY, const MappingParameters &_para, 
		const cv::Mat &WInOneColumn, const cv::Mat &W, cv::Mat &resX, cv::Mat &resY);
};

