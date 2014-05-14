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
	
	//for MUCT test set
	
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
			string field, filename;
			
			// get image filename
			getline(lineStream, filename, seperator);

			list<double> X, Y;

			bool isX = false, isIll = false;
			while(getline(lineStream, field, seperator)){
				double tmp;
				if(isX)
					X.push_back(tmp = string2Double(field));
				else
					Y.push_back(tmp = string2Double(field));

				if(tmp == 0){
					isIll = true;
					break;
				}

				isX = !isX;
			}

			if(!isIll){
				lShapesY.push_back(list2Vec(Y));
				lShapesX.push_back(list2Vec(X));
				imageFilenames.push_back(filename);
			}
		}

		shapesX = list2Mat(lShapesX);
		shapesY = list2Mat(lShapesY);

		// load images
		int count = 0;
		for(list<string>::iterator iter = imageFilenames.begin(); iter != imageFilenames.end(); iter++){
			string &imageFilename = *iter;
			string imagePath = imagesDir + imageFilename + ".jpg";
			cv::Mat image;// = cv::imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
			loadImage(imagePath, image);
			images.push_back(image);

			cout << count++ << " : "<< imageFilename << endl;
		}
		
	} else {
		// fail to open file
		cerr << "Cannot open .csv file : " << filename << endl;
	}
}

char FileManager::getString(FILE *fin, const char sperator, string &res){
	char ch;

	while (ch = fgetc(fin), ch != EOF && (isspace(ch) || ch == sperator));
	res = ch;
	if(ch == EOF) return EOF;

	while(ch = fgetc(fin), !isspace(ch) && ch != sperator){
		res.push_back(ch);
	}
	return ch;
}

char FileManager::getDouble(FILE *fin, double &res){
	char ch;

	while(ch = fgetc(fin), !isdigit(ch) && ch != EOF) ;
	
	if(ch == EOF) return EOF;

	res = ch - 48;

	while(ch = fgetc(fin), isdigit(ch)){
		res = res * 10 + ch - 48;
	}

	if(ch != '.') return ch;

	double frac = 0.1;
	while(ch = fgetc(fin), isdigit(ch)){
		res += (ch - 48) * frac;
		frac *= 0.1;
	}
	return ch;
}

void FileManager::jumpToNextLine(FILE *fin){
	char ch;
	while(ch = fgetc(fin), ch != '\n' && ch != EOF);
}

void FileManager::loadMUCTDataset(const string &filename, const string &imagesDir,
	cv::Mat &shapesX, cv::Mat &shapesY, vector<cv::Mat> &images){
	
	//for MUCT test set _ fast
	
	FILE *fin;
	fopen_s(&fin, filename.c_str(), "r");

	list<string> imageFilenames;
	list<cv::Mat> lShapesX;
	list<cv::Mat> lShapesY;

	//ignore the first line
	jumpToNextLine(fin);
	
	string imgFilename;
	char ch;
	//int count = 0;
	while(ch = getString(fin, ',', imgFilename), ch != EOF){
		list<double> X, Y;
		double x, y;
		bool isIll = false, endOfLine = false;
		while(!endOfLine && getDouble(fin, y) != EOF){
			endOfLine = getDouble(fin, x) == '\n';
			if(x == 0 && y == 0){
				isIll = true;
				jumpToNextLine(fin);
				break;
			}

			X.push_back(x);
			Y.push_back(y);
		}
		if(!isIll){
			//cout << count++ << " : " << imgFilename << endl;
			lShapesX.push_back(list2Vec(X));
			lShapesY.push_back(list2Vec(Y));
			imageFilenames.push_back(imgFilename);
		}
	}

	fclose(fin);

	shapesX = list2Mat(lShapesX);
	shapesY = list2Mat(lShapesY);

	// load images
	for(list<string>::iterator iter = imageFilenames.begin(); iter != imageFilenames.end(); iter++){
		string &imageFilename = *iter;
		string imagePath = imagesDir + imageFilename + ".jpg";
		// this mistake causes bad results
		cv::Mat image;// = cv::imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
		loadImage(imagePath, image);
		images.push_back(image);
	}
}

void FileManager::loadImage(const string &filename, cv::Mat &image){
	image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	image.convertTo(image, CV_64F);	//very important!
}