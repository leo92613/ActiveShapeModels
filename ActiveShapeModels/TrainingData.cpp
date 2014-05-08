#include "TrainingData.h"

TrainingData::TrainingData(void)
{
}

TrainingData::TrainingData(const string &csvFilename, const string &imagesDir){
	cerr << "initialize training data" << endl;
	loadDataAndImagesFromCSV(csvFilename, imagesDir);
	generateWAndWInOneColumn();
	generateGradientImages();
	generateLocalFeatures();
	generateAlignedShapes();
	generatePCAShapeModel();
	cerr << "initializing compeleted" << endl;
}


TrainingData::~TrainingData(void)
{
}

void TrainingData::loadDataAndImagesFromCSV(const string &csvFilename, const string &imagesDir){
	cerr << "load training data and training images from csv file" << endl;

	FileManager fileManager;
	fileManager.loadDataAndImagesFromCSV(csvFilename, imagesDir, trainingShapesX, trainingShapesY, trainingImages);
	
	//debug
	//cout << trainingShapesX.col(0) << endl;
	//system("pause");
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
	cerr << "generate mat W and vec WInOneColumn" << endl;

	const int numberOfShapes = trainingShapesX.cols;
	const int numberOfPoints = trainingShapesX.rows;

	WInOneColumn = cv::Mat(numberOfPoints, 1, CV_64F);
	W = cv::Mat(numberOfPoints, numberOfPoints, CV_64F, cv::Scalar::all(0));

	for(int k = 0; k < numberOfPoints; k++){
		W.at<double>(k, k) = WInOneColumn.at<double>(k) = getWk(k);
	}
}

void TrainingData::generateGradientImages(){
	cerr << "generate gradient images" << endl;

	for(std::vector<cv::Mat>::iterator iter = trainingImages.begin(); iter != trainingImages.end(); iter++){
		cv::Mat gradX, gradY, grad;
		cv::Sobel(*iter, gradX, (*iter).depth(), 1, 0);
		cv::Sobel(*iter, gradY, (*iter).depth(), 0, 1);
		cv::convertScaleAbs(gradX, gradX);
		cv::convertScaleAbs(gradY, gradY);
		cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, grad);
		
		grad.convertTo(grad, CV_64F);
		//normalize(grad, grad, 0, 255, cv::NORM_MINMAX, -1);
		gradientImages.push_back(grad);

		//debug
		//cout << grad.col(1) << endl;
		//cv::namedWindow("Check graident image", CV_WINDOW_AUTOSIZE);
		//cv::imshow("Check graident image", grad);
		//cv::waitKey(0);
		//end debug
	}
}

void TrainingData::generateLocalFeatures(){
	cerr << "generate local features" << endl;

	const int numberOfPoints = trainingShapesX.rows;

	for(int i = 0; i < numberOfPoints; i++){
		LocalFeature _localFeature;
		_localFeature.computeLocalFeature(trainingShapesX, 
			trainingShapesY, 
			gradientImages, 
			i);
		localFeatures.push_back(_localFeature);
	}
}

void TrainingData::generatePCAShapeModel(){
	cerr << "generate PCA shape model" << endl;
	pcaShapeModel.generateBases(alignedShapesX, alignedShapesY, meanAlignedShapesX, meanAlignedShapesY);
}

void TrainingData::generateAlignedShapes(){
	cerr << "generate aligned shapes" << endl;

	AlignShape alignShape;
	alignShape.alignTrainingShapes(trainingShapesX, 
		trainingShapesY, 
		WInOneColumn, 
		W, 
		c_alignIterationTimeThreshold, 
		c_alignConvergencyThreshold, 
		alignedShapesX, 
		alignedShapesY, 
		meanAlignedShapesX, 
		meanAlignedShapesY);

	//debug
	//cout << meanAlignedShapesX << endl;
	//end debug
}

void TrainingData::findBestShifts(const cv::Mat &shapeX, const cv::Mat &shapeY, const cv::Mat &gradientImage,
								  cv::Mat &shiftsX, cv::Mat &shiftsY){
	const int numberOfPoints = shapeX.rows;
	
	shiftsX = cv::Mat(numberOfPoints, 1, CV_64F);
	shiftsY = cv::Mat(numberOfPoints, 1, CV_64F);

	for(int i = 0; i < numberOfPoints; i++){
		localFeatures[i].findBestShift(shapeX, shapeY, gradientImage, i, 
			shiftsX.at<double>(i, 0), shiftsY.at<double>(i, 0));
	}

	//debug
	cout << "shiftX : " << endl;
	cout << shiftsX << endl;
	cout << "shiftY : " << endl;
	cout << shiftsY << endl;
	//end debug
}

void TrainingData::findBestDeforming(const cv::Mat &X0, const cv::Mat &Y0, const cv::Mat &sX, const cv::Mat &sY,
		 const MappingParameters &_para, cv::Mat &resX, cv::Mat &resY){
	pcaShapeModel.findBestDeforming(X0, Y0, sX, sY, _para, WInOneColumn, W, resX, resY);
}