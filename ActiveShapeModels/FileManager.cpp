#include "FileManager.h"

FileManager::FileManager(void)
{
}


FileManager::~FileManager(void)
{
}

void FileManager::getFilenamesByPathAndExtension(string path, string extension, list<string> &filenames){
	cv::Directory dir;

	list<string> directories;
	directories.push_back(path);
	
	for(list<string>::iterator iter = directories.begin(); iter != directories.end(); iter++){
		vector<string> _directories = dir.GetListFolders(*iter);
		directories.insert(directories.end(), _directories.begin(), _directories.end());

		vector<string> _filenames = dir.GetListFiles(*iter, extension, true);
		filenames.insert(filenames.end(), _filenames.begin(), _filenames.end());
	}
}

double FileManager::string2Double(const string &s){
	istringstream stream(s);
	double result;
	stream >> result;
	return result;
}

cv::Mat FileManager::list2Vec(list<double> &L){
	const int rows = L.size();
	cv::Mat vec(rows, 1, CV_64F);

	list<double>::iterator iter = L.begin();
	for(int nrow = 0; iter != L.end(); iter++, nrow++){
		vec.at<double>(nrow) = *iter;
	}

	return vec;
}

cv::Mat FileManager::list2Mat(list<cv::Mat> &L){
	list<cv::Mat>::iterator iter = L.begin();
	cv::Mat &vec0 = *iter;
	
	const int rows = vec0.rows;
	const int cols = L.size();

	cv::Mat mat(rows, cols, CV_64F);

	for(int ncol = 0; iter != L.end(); iter++, ncol++){
		cv::Mat &vec = *iter;
		vec.copyTo(mat.col(ncol));
	}

	return mat;
}

void FileManager::loadDataAndImagesFromCSV(const string &filename, const string &imagesDir,
	cv::Mat &shapesX, cv::Mat &shapesY, vector<cv::Mat> &images){
	ifstream fin(filename.c_str());

	if(fin.is_open()){
		// ignore the first line
		string line; // line is the content of each line
		const char seperator = ',';
		
		list<string> imageFilenames;
		list<cv::Mat> lShapesX;
		list<cv::Mat> lShapesY;

		getline(fin, line);

		while(getline(fin, line)){
			istringstream lineStream(line);
			string field;
			
			// get image filename
			getline(lineStream, field, seperator);
			imageFilenames.push_back(field);

			//debug
			//cout << field << endl;
			//end debug

			list<double> X, Y;

			bool isX = true;
			while(getline(lineStream, field, seperator)){
				if(isX)
					X.push_back(string2Double(field));
				else
					Y.push_back(string2Double(field));

				isX = !isX;
			}
			
			//debug
			//cout << list2Vec(Y) << endl;
			//end debug

			lShapesX.push_back(list2Vec(X));
			lShapesY.push_back(list2Vec(Y));
		}

		shapesX = list2Mat(lShapesX);
		shapesY = list2Mat(lShapesY);

		//debug
		//cout << shapesX.col(0) << endl;
		//cout << shapesY.col(0) << endl;
		//end debug

		// load images
		for(list<string>::iterator iter = imageFilenames.begin(); iter != imageFilenames.end(); iter++){
			string &imageFilename = *iter;
			string imagePath = imagesDir + imageFilename + ".jpg";
			cv::Mat image = cv::imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
			//image.convertTo(image, CV_64F);
			//debug
			//cout << imagePath << endl;
			//cv::namedWindow("Check input image", CV_WINDOW_AUTOSIZE);
			//cv::normalize(image, image, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			//cv::imshow("Check input image", image);
			//cv::waitKey(0);
			//end debug

			images.push_back(image);
		}
		
	} else {
		// fail to open file
		cerr << "Cannot open .csv file : " << filename << endl;
	}
}

void FileManager::loadImage(const string &filename, cv::Mat &image){
	image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	image.convertTo(image, CV_64F);
}