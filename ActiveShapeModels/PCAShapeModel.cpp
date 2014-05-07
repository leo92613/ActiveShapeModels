#include "PCAShapeModel.h"


PCAShapeModel::PCAShapeModel(void)
{
}


PCAShapeModel::~PCAShapeModel(void)
{
}

void PCAShapeModel::mergeXY(const cv::Mat &X, const cv::Mat &Y, cv::Mat &XY){
	const int rows = X.rows, cols = X.cols;
	XY = cv::Mat(rows * 2, cols, X.type());
	
	XY(cv::Range(0, rows - 1), cv::Range(0, cols - 1)) = X;
	XY(cv::Range(rows, rows * 2 - 1), cv::Range(0, cols -1)) = Y;
}

void PCAShapeModel::splitXY(const cv::Mat &XY, cv::Mat &X, cv::Mat &Y){
	const int rows = XY.rows;
	int rows2 = rows / 2;
	X = XY.rowRange(cv::Range(0, rows2 - 1));
	Y = XY.rowRange(cv::Range(rows2, rows - 1));
}

void PCAShapeModel::generateBases(const cv::Mat &alignedShapesX, const cv::Mat &alignedShapesY, 
				   const cv::Mat &meanShapeX, const cv::Mat &meanShapeY){
	
	const int numberOfShapes = alignedShapesX.cols;
	const int numberOfPoints = alignedShapesX.rows;

	cv::Mat shapesXY;
	mergeXY(alignedShapesX, alignedShapesY, shapesXY);

	cv::Mat meanShapeXY;
	mergeXY(meanShapeX, meanShapeY, meanShapeXY);

	PCAShapeModel::meanShapeX = meanShapeX;
	PCAShapeModel::meanShapeY = meanShapeY;

	pca = cv::PCA(shapesXY,
				meanShapeXY,
				CV_PCA_DATA_AS_COL,
				c_retainedVariance);
}

void PCAShapeModel::findBestDeforming(const cv::Mat &X0, const cv::Mat &Y0,
	const cv::Mat &sX, const cv::Mat &sY, const MappingParameters &_para, 
	cv::Mat &resX, cv::Mat &resY){
	
	MappingParameters para = _para;
	para.inverse();
	
	cv::Mat X, Y;
	para.getAlignedXY(X0, Y0, X, Y, -1.0);
	X += (1.0 / para.scale) * sX;
	Y += (1.0 / para.scale) * sY;

	cv::Mat XY;
	mergeXY(X, Y, XY);

	cv::Mat b;
	pca.project(XY, b);
	pca.backProject(b, XY);

	splitXY(XY, resX, resY);
}