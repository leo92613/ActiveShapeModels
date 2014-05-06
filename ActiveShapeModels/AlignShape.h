#pragma once

#include <opencv2\core\core.hpp>
#include "MappingParameters.h"

class AlignShape
{
public:
	AlignShape(void);
	~AlignShape(void);

	MappingParameters findBestMapping(const cv::Mat &Ax, const cv::Mat &Ay, 
		const cv::Mat &Bx, const cv::Mat By, const cv::Mat WInOneColumn, const cv::Mat W);
};

