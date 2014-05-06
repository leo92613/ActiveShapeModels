#include "LocalFeature.h"


LocalFeature::LocalFeature(void)
{
}


LocalFeature::~LocalFeature(void)
{
}

void LocalFeature::computeLocalFeature(const cv::Mat &trainingShapesX, const cv::Mat &trainingShapesY, 
									   const cv::Mat &gradientImages, const int curr){
	const int numberOfShapes = trainingShapesX.cols;
	const int numberOfPoints = trainingShapesX.rows;

	const int prev = (curr - 1 + numberOfPoints) % numberOfPoints;
	const int next = (curr + 1) % numberOfPoints;

	cv::Mat featureVectors(_featureSize * 2 + 1, numberOfShapes, CV_64F);

	for(int i = 0; i < numberOfShapes; i++){
		double x0 = trainingShapesX.at<double>(curr, i);
		double x1 = trainingShapesX.at<double>(prev, i) - x0;
		double x2 = trainingShapesX.at<double>(next, i) - x0;
		double y0 = trainingShapesY.at<double>(curr, i);
		double y1 = trainingShapesY.at<double>(prev, i) - y0;
		double y2 = trainingShapesY.at<double>(next, i) - y0;

		double dx = x1 + x2, dy = y1 + y2;
		double _normd = sqrt(dx * dx + dy * dy);
		dx /= _normd; dy /= _normd;

		for(int scale = -_featureSize; scale <= _featureSize; scale++){
			int _x = x0 + dx * scale;
			int _y = y0 + dy * scale;
			featureVectors.at<double>(scale + _featureSize, i) = gradientImages.at<double>(_x, _y);
		}
	}

	cv::calcCovarMatrix(featureVectors, covar, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);
}

void LocalFeature::findBestShift(const cv::Mat &shapeX, const cv::Mat &shapeY, const cv::Mat &gradImage, 
				   const int curr, cv::Mat &shiftX, cv::Mat &shiftY){
	
	const int numberOfPoints = shapeX.rows;
	const int prev = (curr - 1 + numberOfPoints) % numberOfPoints;
	const int next = (curr + 1) % numberOfPoints;
	shiftX = cv::Mat(numberOfPoints, 1, CV_64F);
	shiftY = cv::Mat(numberOfPoints, 1, CV_64F);

	for(int i = 0; i < numberOfPoints; i++){
		cv::Mat features = cv::Mat(_featureSize * 4 + 1, 1, CV_64F);

		double x0 = shapeX.at<double>(curr, i);
		double x1 = shapeX.at<double>(prev, i) - x0;
		double x2 = shapeX.at<double>(next, i) - x0;
		double y0 = shapeY.at<double>(curr, i);
		double y1 = shapeY.at<double>(prev, i) - y0;
		double y2 = shapeY.at<double>(next, i) - y0;

		double dx = x1 + x2, dy = y1 + y2;
		double _normd = sqrt(dx * dx + dy * dy);
		dx /= _normd; dy /= _normd;

		for(int scale = -2 * _featureSize; scale <= 2 * _featureSize; scale++){
			int _x = x0 + dx * scale;
			int _y = y0 + dy * scale;
			features.at<double>(scale + 2 * _featureSize, i) = gradImage.at<double>(_x, _y);
		}

		double maxDist = -1e60;
		int maxDist_x = -1, maxDist_y = -1;
		for(int scale = -2 * _featureSize; scale <= _featureSize; scale++){
			int _x = x0 + dx * scale;
			int _y = y0 + dy * scale;

			cv::Mat featureAtXY = features(cv::Range(0, 0), cv::Range(scale, scale + _featureSize - 1));
			double _dist = cv::Mahalanobis(featureAtXY, mean, covar);
			if(_dist > maxDist){
				maxDist = _dist;
				maxDist_x = _x;
				maxDist_y = _y;
			}
		}
		shiftX.at<double>(i, 0) = maxDist_x - shapeX.at<double>(i, 0);
		shiftY.at<double>(i, 0) = maxDist_y - shapeY.at<double>(i, 0);
	}
}