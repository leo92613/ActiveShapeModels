#include "ActiveShapeModels.h"


ActiveShapeModels::ActiveShapeModels(void)
{
}


ActiveShapeModels::~ActiveShapeModels(void)
{
}

void ActiveShapeModels::creatInitialShape(){
}

void ActiveShapeModels::iterationSearch(TrainingData &trainingData){

	cv::Mat shiftX, shiftY;
	cv::Mat lastShapeX, lastShapeY;

	creatInitialShape();
	for(int i = 0; i < c_asmSearchThreshold; i++){
		shapeX.copyTo(lastShapeX);
		shapeY.copyTo(lastShapeY);

		trainingData.findBestShifts(lastShapeX, lastShapeY, gradiantImage, shiftX, shiftY);
		
		MappingParameters para;
		AlignShape alignShape;
		para = alignShape.findBestMapping(lastShapeX + shiftX, lastShapeY + shiftY, 
			shapeX, shapeY, trainingData.WInOneColumn, trainingData.W);
		
		//para.getAlignedXY(lastShapeX, lastShapeY, shapeX, shapeY);

		trainingData.pcaShapeModel.findBestDeforming(lastShapeX, lastShapeY, shiftX, shiftY,
			para, shapeX, shapeY);
	}
}