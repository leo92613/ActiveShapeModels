#pragma once
#include <string>
#include <list>
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\ml\ml.hpp>
#include <iostream>
#include <fstream>

using namespace std;

class FileManager
{
public:
	FileManager(void);
	~FileManager(void);

	void getFilenamesByPathAndExtension(string path, string extension, list<string> &filenames);

	double string2Double(const string &s);
	cv::Mat list2Vec(list<double> &L);
	cv::Mat list2Mat(list<cv::Mat> &L);

	void loadImage(const string &filename, cv::Mat &image);
	void loadDataAndImagesFromCSV(const string &filename,	const string &imagesDir,
		cv::Mat &shapesX, cv::Mat &shapesY, vector<cv::Mat> &images);

	string getPathFromUser();
	string getListFromUser();
};

