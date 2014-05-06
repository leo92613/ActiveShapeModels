#include "AlignShape.h"


AlignShape::AlignShape(void)
{
}


AlignShape::~AlignShape(void)
{
}

MappingParameters findBestMapping(const cv::Mat &Ax, const cv::Mat &Ay, 
	const cv::Mat &Bx, const cv::Mat &By, const cv::Mat &WInOneColumn, const cv::Mat &W){
	
	cv::Mat _X1, _Y1, _X2, _Y2;
	double X1, Y1, X2, Y2;
	cv::Mat WInOneColumnT = WInOneColumn.t();
	
	_X1 = WInOneColumnT * Ax;
	_Y1 = WInOneColumnT * Ay;
	_X2 = WInOneColumnT * Bx;
	_Y2 = WInOneColumnT * By;

	X1 = _X1.at<double>(0, 0);
	Y1 = _Y1.at<double>(0, 0);
	X2 = _X2.at<double>(0, 0);
	Y2 = _Y2.at<double>(0, 0);

	cv::Mat _WBxBx, _WByBy, _WAxBx, _WAxBy, _WAyBx, _WAyBy;
	double WBxBx, WByBy, WAxBx, WAxBy, WAyBx, WAyBy;
	double C1, C2, Z;
	_WBxBx = Bx.t() * W * Bx;
	_WByBy = By.t() * W * By;
	
	_WAxBx = Ax.t() * W * Bx;
	_WAxBy = Ax.t() * W * By;

	_WAyBx = Ay.t() * W * Bx;
	_WAyBy = Ay.t() * W * By;

	WBxBx = _WBxBx.at<double>(0, 0);
	WByBy = _WByBy.at<double>(0, 0);
	WAxBx = _WAxBx.at<double>(0, 0);
	WAxBy = _WAxBy.at<double>(0, 0);
	WAyBx = _WAyBx.at<double>(0, 0);
	WAyBy = _WAyBy.at<double>(0, 0);

	Z = WBxBx + WByBy;
	C1 = WAxBx + WAyBy;
	C2 = WAyBx - WAxBy;

	cv::Mat _sumW(WInOneColumn.size(), CV_64F, cv::Scalar::all(1));
	_sumW = WInOneColumnT * _sumW;
	double sumW = _sumW.at<double>(0, 0);
	
	//solve Ax = b
	cv::Mat _A = (cv::Mat_<double>(4,4) << X2, -Y2,  sumW,  0,
										Y2,  X2,  0,  sumW,
										 Z,   0, X2, Y2,
										 0,   Z,-Y2, X2);
	cv::Mat _b = (cv::Mat_<double>(4,1) << X1, Y1, C1, C2);
	cv::Mat _x;
	
	cv::solve(_A, _b, _x);
	//_x = [ax, ay, tx, ty]T

	MappingParameters result;
	double _ax = _x.at<double>(0, 0), _ay = _x.at<double>(1, 0);
	double _tx = _x.at<double>(2, 0), _ty = _x.at<double>(3, 0);

	result.scale = sqrt(_ax * _ax + _ay * _ay);
	result.rotation = atan2(_ay, _ax);	//use atan2 to get a angle between -pi .. pi
	result.translationX = _tx;
	result.translationY = _ty;

	return result;
}