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
	//debug 
	//cv::namedWindow("searchingImage", CV_WINDOW_AUTOSIZE);
	//cv::imshow("searchingImage", image);
	//cv::waitKey(0);
	//end debug
}

void ActiveShapeModels::generateGradiantImage(){
	cv::Mat gradX, gradY, grad;
	cv::Sobel(image, gradX, image.depth(), 1, 0);
	cv::Sobel(image, gradY, image.depth(), 0, 1);
	cv::convertScaleAbs(gradX, gradX);
	cv::convertScaleAbs(gradY, gradY);
	cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, grad);
		
	grad.convertTo(gradiantImage, CV_64F);

	//debug
	//cv::namedWindow("gradSearchingImage", CV_WINDOW_AUTOSIZE);
	//cv::imshow("gradSearchingImage", gradiantImage);
	//cv::waitKey(0);
	//end debug
}

void ActiveShapeModels::creatInitialShape(TrainingData &trainingData){
	cerr << "creat initial shape" << endl;
	trainingData.meanAlignedShapesX.copyTo(shapeX);
	trainingData.meanAlignedShapesY.copyTo(shapeY);
}

void ActiveShapeModels::iterationSearch(TrainingData &trainingData){
	cerr << "start iterationSearch" << endl;

	cv::Mat shiftX, shiftY;
	cv::Mat lastShapeX, lastShapeY;

	creatInitialShape(trainingData);
	//debug
	ResultProcessor resultProcessor;
	resultProcessor.showResultImage(shapeX, shapeY, image, "Search result");
	//end debug

	for(int i = 0; i < c_asmSearchThreshold; i++){
		shapeX.copyTo(lastShapeX);
		shapeY.copyTo(lastShapeY);

		trainingData.findBestShifts(lastShapeX, lastShapeY, gradiantImage, shiftX, shiftY);
		
		//debug
		cv::Mat tmpMat;
		gradiantImage.convertTo(tmpMat, CV_64F);
		resultProcessor.showResultImage(lastShapeX + shiftX, lastShapeY +shiftY, tmpMat, "shift result");
		//end debug

		MappingParameters para;
		AlignShape alignShape;
		para = alignShape.findBestMapping(lastShapeX + shiftX, lastShapeY + shiftY, 
			lastShapeX, lastShapeY, trainingData.WInOneColumn, trainingData.W);
		
		//debug
		//cout << para << endl;
		cv::Mat _sX, _sY;
		para.getAlignedXY(lastShapeX, lastShapeY, _sX, _sY);
		resultProcessor.showResultImage(_sX, _sY, image, "aligned result");
		//end debug


		//trainingData.findBestDeforming(lastShapeX, lastShapeY, shiftX, shiftY,
		//	para, shapeX, shapeY);
		
		//debug
		shapeX = _sX;
		shapeY = _sY;
		resultProcessor.showResultImage(shapeX, shapeY, image, "Search result");
		//end debug
	}
}