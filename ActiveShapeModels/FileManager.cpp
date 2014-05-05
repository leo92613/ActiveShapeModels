#include "FileManager.h"
#include "opencv2\opencv.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"

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

