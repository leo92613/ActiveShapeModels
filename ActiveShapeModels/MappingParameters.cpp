#include "MappingParameters.h"
#include <cmath>

MappingParameters::MappingParameters(void)
{
	scale = rotation = translationX = translationY = 0;
}


MappingParameters::~MappingParameters(void)
{
}

void MappingParameters::caculateNewCoordinates(double x, double y, double &resX, double &resY){
	double SCosR = scale * cos(rotation), SSinR = scale * sin(rotation);
	resX = x * SCosR - y * SSinR + translationX;
	resY = x * SSinR + y * SCosR + translationY;
}

void MappingParameters::getMappingMatrix(cv::Mat &mappingMatrix){
	//return a matrix like
	//			[ScosR	-SsinR]
	//			[SsinR	 ScosR]
	double SCosR = scale * cos(rotation), SSinR = scale * sin(rotation);
	mappingMatrix = (cv::Mat_<double>(2, 2) << SCosR, -SSinR,
												SSinR, SCosR);
}

void MappingParameters::getTranslationMatrix(cv::Mat &translationMatrix){
	//return a matrix like
	//					[tx 0]
	//					[0 ty]
	translationMatrix = (cv::Mat_<double>(2, 2) << translationX, 0,
													0, translationY);
}
