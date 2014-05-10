#pragma once

#include <opencv2\core\core.hpp>
#include <iostream>

using std::ostream;

class MappingParameters
{
public:
	MappingParameters(void);
	~MappingParameters(void);

	friend ostream& operator << (ostream &os, const MappingParameters &para);

	double scale, rotation, translationX, translationY;
	void caculateNewCoordinates(double x, double y, double &resX, double &resY);

	void inverse();

	void getMappingMatrix(cv::Mat &mappingMatrix);
	void getTranslationMatrix(cv::Mat &translationMatrix);
	void getAlignedXY(const cv::Mat &shapeX, const cv::Mat &shapeY, 
		cv::Mat &newShapeX, cv::Mat &newShapeY);

	void getAlignedXY2(const cv::Mat &shapeX, const cv::Mat &shapeY, 
		const cv::Mat &sX, const cv::Mat &sY, cv::Mat &newShapeX, cv::Mat &newShapeY);
};