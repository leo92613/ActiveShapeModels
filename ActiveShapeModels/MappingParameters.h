#pragma once
class MappingParameters
{
public:
	MappingParameters(void);
	~MappingParameters(void);

	double scale, rotation, translationX, translationY;
	void caculateNewCoordinates(double x, double y, double &resX, double &resY);
};

