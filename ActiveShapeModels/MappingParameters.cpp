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