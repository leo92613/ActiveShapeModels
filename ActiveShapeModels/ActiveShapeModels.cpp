#include "ActiveShapeModels.h"


ActiveShapeModels::ActiveShapeModels(void)
{
}


ActiveShapeModels::~ActiveShapeModels(void)
{
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

	for(int i = 0; i < c_asmSearchThreshold; i++){
		shapeX.copyTo(lastShapeX);
		shapeY.copyTo(lastShapeY);

		trainingData.findBestShifts(lastShapeX, lastShapeY, gradiantImage, shiftX, shiftY);
		
		MappingParameters para;
		AlignShape alignShape;
		para = alignShape.findBestMapping(lastShapeX + shiftX, lastShapeY + shiftY, 
			shapeX, shapeY, trainingData.WInOneColumn, trainingData.W);

		trainingData.findBestDeforming(lastShapeX, lastShapeY, shiftX, shiftY,
			para, shapeX, shapeY);
	}
}