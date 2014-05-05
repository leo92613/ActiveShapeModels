#include <iostream>
#include "FileManager.h"
#include <list>
#include <string>

using namespace std;

int main(){
	FileManager fileManager;
	list<string> filenames;
	string path = "D:\\Eigenfaces_for_Recognition\\att_faces_png";
	fileManager.getFilenamesByPathAndExtension(path, "*.png", filenames);

	for(list<string>::iterator iter = filenames.begin(); iter != filenames.end(); iter++){
		cout << *iter << endl;
	}

	return 0;
}