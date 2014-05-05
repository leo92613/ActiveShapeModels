#pragma once
#include <string>
#include <list>

using namespace std;

class FileManager
{
public:
	FileManager(void);
	~FileManager(void);

	void getFilenamesByPathAndExtension(string path, string extension, list<string> &filenames);
	void getFilenameWithExtensionFromUser(string extension);
	void getFilenamesByList();

	string getPathFromUser();
	string getListFromUser();
};

