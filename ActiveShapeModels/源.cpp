#include "FileManager.h"
#include "TrainingData.h"
#include "ActiveShapeModels.h"
#include "ResultProcessor.h"
#include <string>

using namespace std;

const string csvFilename("D:\\ActiveShapeModels\\TestData\\MUCT\\MUCT2.csv");
const string imagesDir("D:\\ActiveShapeModels\\TestData\\MUCT\\");
const string searchedImageFilename("D:\\ActiveShapeModels\\TestData\\MUCT\\i002se-mn.jpg");

int main(){

	TrainingData trainingData(csvFilename, imagesDir);
	ActiveShapeModels ASM(searchedImageFilename);
	ASM.iterationSearch(trainingData);

	//ResultProcessor resultProcessor;
	//resultProcessor.showResultImage(ASM.shapeX, ASM.shapeY, ASM.image, "Search result");
	return 0;
}