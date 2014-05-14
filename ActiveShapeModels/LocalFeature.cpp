#include "LocalFeature.h"

//debug
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
using std::cout;
using std::endl;
#include "ResultProcessor.h"
//end debug

LocalFeature::LocalFeature(void)
{
}


LocalFeature::~LocalFeature(void)
{
}

void LocalFeature::caculateDxDy(const double x1, const double y1,
		const double x2, const double y2, double &dx, double &dy){
	dx = -y1 + y2;
	dy = x1 - x2;
	double normd = sqrt(dx * dx + dy * dy);
	dx /= normd; dy /= normd;
	if(normd == 0){
		dx = 1; dy = 0;
	}
}

void LocalFeature::computeLocalFeature(const cv::Mat &trainingShapesX, const cv::Mat &trainingShapesY, 
									   const std::vector<cv::Mat> &gradientImages, const int curr){
	const int numberOfShapes = trainingShapesX.cols;
	const int numberOfPoints = trainingShapesX.rows;

	const int prev = (curr - 1 + numberOfPoints) % numberOfPoints;
	const int next = (curr + 1) % numberOfPoints;

	cv::Mat featureVectors(c_featureSize * 2 + 1, numberOfShapes, CV_64F, cv::Scalar::all(0));

	for(int i = 0; i < numberOfShapes; i++){
  		double x0 = trainingShapesX.at<double>(curr, i);
		double x1 = trainingShapesX.at<double>(prev, i);
		double x2 = trainingShapesX.at<double>(next, i);
		double y0 = trainingShapesY.at<double>(curr, i);
		double y1 = trainingShapesY.at<double>(prev, i);
		double y2 = trainingShapesY.at<double>(next, i);

		double dx, dy;
		caculateDxDy(x1, y1, x2, y2, dx, dy);

		for(int scale = -c_featureSize; scale <= c_featureSize; scale++){
			double _x = x0 + dx * scale;
			double _y = y0 + dy * scale;

			if(_x < 0 || _x >= gradientImages[i].rows || _y < 0 || _y >= gradientImages[i].cols) continue;

			featureVectors.at<double>(scale + c_featureSize, i) = gradientImages[i].at<double>(_x, _y);
		}

		cv::normalize(featureVectors.col(i), featureVectors.col(i), 1, 0, cv::NORM_L1);
	}

	cv::calcCovarMatrix(featureVectors, covar, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);
	cv::invert(covar, icovar, cv::DECOMP_SVD);
}

void LocalFeature::findBestShift(const cv::Mat &shapeX, const cv::Mat &shapeY, const cv::Mat &gradientImage,
				   const int curr, double &shiftX, double &shiftY){
	
	const int numberOfPoints = shapeX.rows;
	const int prev = (curr - 1 + numberOfPoints) % numberOfPoints;
	const int next = (curr + 1) % numberOfPoints;

	double x0 = shapeX.at<double>(curr, 0);
	double x1 = shapeX.at<double>(prev, 0);
	double x2 = shapeX.at<double>(next, 0);
	double y0 = shapeY.at<double>(curr, 0);
	double y1 = shapeY.at<double>(prev, 0);
	double y2 = shapeY.at<double>(next, 0);

	double dx, dy;
	caculateDxDy(x1, y1, x2, y2, dx, dy);

	int idShift = c_shiftWindowSize + c_featureSize;

	cv::Mat features = cv::Mat(idShift * 2 + 1, 1, CV_64F, cv::Scalar::all(0));

	for(int scale = - idShift; scale <= idShift; scale++){
		double _x = x0 + dx * scale;
		double _y = y0 + dy * scale;

		if(_x < 0 || _x >= gradientImage.rows || _y < 0 || _y >= gradientImage.cols) continue;

		features.at<double>(scale + idShift) = gradientImage.at<double>(_x, _y);
	}

	double minDist = 1e60;
	double minDist_x = x0, minDist_y = y0 ;

	for(int scale =  - c_shiftWindowSize; scale <= c_shiftWindowSize; scale++){
		double _x = x0 + dx * scale;
		double _y = y0 + dy * scale;

		if(_x < 0 || _x >= gradientImage.rows || _y < 0 || _y >= gradientImage.cols) continue;
		//边界还没加上窗口部分

		cv::Mat featureAtXY = features.rowRange(scale - c_featureSize + idShift, scale + c_featureSize + idShift + 1).clone();
	
		cv::normalize(featureAtXY, featureAtXY, 1, 0, cv::NORM_L1);

		double _dist = cv::Mahalanobis(featureAtXY, mean, icovar);
		if(_dist < minDist){
			minDist = _dist;
			minDist_x = _x;
			minDist_y = _y;
		}
	}

	shiftX = minDist_x - x0;
	shiftY = minDist_y - y0;
}