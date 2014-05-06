#pragma once

#include <opencv2\core\core.hpp>

class MappingParameters
{
public:
	MappingParameters(void);
	~MappingParameters(void);

	double scale, rotation, translationX, translationY;
	void caculateNewCoordinates(double x, double y, double &resX, double &resY);

	void getMappingMatrix(cv::Mat &mappingMatrix);
	void getTranslationMatrix(cv::Mat &translationMatrix);
	void getAlignedXY(const cv::Mat &shapeX, const cv::Mat &shapeY, cv::Mat &newShapeX, cv::Mat &newShapeY);
};

