#include "TrainingData.h"

TrainingData::TrainingData(void)
{
}


TrainingData::~TrainingData(void)
{
}


double TrainingData::getWk(int k){
	const int numberOfShapes = trainingShapesX.cols;
	const int numberOfPoints = trainingShapesX.rows;

	cv::Mat D(1, numberOfShapes, CV_64F);

	cv::Mat allOneMat(numberOfPoints, 1, CV_64F, cv::Scalar::all(1));

	for(int l = 0; l < numberOfShapes; l++){
		cv::Mat dX = trainingShapesX.col(l) - allOneMat * trainingShapesX.at<double>(k, l);
		cv::Mat dY = trainingShapesY.col(l) - allOneMat * trainingShapesY.at<double>(k, l);
		double _distX = cv::norm(dX), _distY = cv::norm(dY);
		double _dist = sqrt(_distX * _distX + _distY * _distY);
		D.at<double>(0, l) = _dist;
	}

	double sum = 0.0;
	for(int l = 0; l < numberOfShapes; l++) sum += D.at<double>(0, l);
	double mean = sum / numberOfShapes;
	D = D - allOneMat * mean;
	double varience = cv::norm(D);
	varience *= varience;
	varience /= (numberOfShapes - 1);
	return 1 / varience;
}

void TrainingData::generateWAndWInOneColumn(){
	
}