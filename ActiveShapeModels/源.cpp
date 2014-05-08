#include "FileManager.h"
#include "TrainingData.h"
#include "ActiveShapeModels.h"
#include <string>

using namespace std;

const string csvFilename("D:\\ActiveShapeModels\\TestData\\MUCT\\MUCT2.csv");
const string imagesDir("D:\\ActiveShapeModels\\TestData\\MUCT\\");

int main(){

	TrainingData trainingData(csvFilename, imagesDir);
	ActiveShapeModels ASM;
	ASM.iterationSearch(trainingData);

	return 0;
}