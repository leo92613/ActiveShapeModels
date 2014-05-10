#include "ActiveShapeModels.h"

//debug
#include "ResultProcessor.h"
//end debug

ActiveShapeModels::ActiveShapeModels(void)
{
}


ActiveShapeModels::~ActiveShapeModels(void)
{
}

ActiveShapeModels::ActiveShapeModels(const string &filename){
	loadImage(filename);
	generateGradiantImage();
}

void ActiveShapeModels::loadImage(const string &filename){
	cerr << "load image for searching" << endl;
	FileManager fileManager;
	fileManager.loadImage(filename, image);
}

void ActiveShapeModels::generateGradiantImage(){
	cv::Mat gradX, gradY, grad;
	cv::Sobel(image, gradX, image.depth(), 1, 0);
	cv::Sobel(image, gradY, image.depth(), 0, 1);
	cv::convertScaleAbs(gradX, gradX);
	cv::convertScaleAbs(gradY, gradY);
	cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, grad);
	
	grad.convertTo(gradiantImage, CV_64F);
}

void ActiveShapeModels::creatInitialShape(TrainingData &trainingData){
	cerr << "creat initial shape" << endl;
	trainingData.meanAlignedShapesX.copyTo(shapeX);
	trainingData.meanAlignedShapesY.copyTo(shapeY);
}

void ActiveShapeModels::iterationSearch(TrainingData &trainingData){
	cerr << "start iterationSearch" << endl;

	cv::Mat shiftX, shiftY;

	creatInitialShape(trainingData);
	//debug
	ResultProcessor resultProcessor;
	resultProcessor.showResultImage(shapeX, shapeY, gradiantImage, "Search result");
	//end debug

	for(int i = 0; i < c_asmSearchThreshold; i++){
		trainingData.findBestShifts(shapeX, shapeY, gradiantImage, shiftX, shiftY);
		
		//debug
		cv::Mat tmpMat;
		gradiantImage.convertTo(tmpMat, CV_64F);
		resultProcessor.showResultImage(shapeX + shiftX, shapeY +shiftY, tmpMat, "shift result");
		//end debug

		MappingParameters para;
		AlignShape alignShape;
		para = alignShape.findBestMapping(shapeX + shiftX, shapeY + shiftY, 
			shapeX, shapeY, trainingData.WInOneColumn, trainingData.W);

		trainingData.findBestDeforming(shapeX, shapeY, shiftX, shiftY, para, shapeX, shapeY);
		
		//debug
		resultProcessor.showResultImage(shapeX, shapeY, image, "Search result");
		//end debug
	}
}