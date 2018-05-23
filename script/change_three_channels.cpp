/*
string dir_path = "./image_9000/";             //original images which include abnomral images
string output_path = "./output/";              //abnormal images which is in three channels
*/

#include <cstdio>  
#include <vector>  
#include <iostream>  
#include <fstream>  
#include <cstring>  
#include <cstdlib>  
#include <cmath>  
#include <algorithm> 
#include <stdio.h>

#include "opencv\cv.h"  
#include "opencv2\core\core.hpp"  
#include "opencv2\highgui\highgui.hpp"  
#include "opencv2\imgproc\imgproc.hpp"  
#include "opencv2\contrib\contrib.hpp"  

using namespace std;
using namespace cv;

int main()
{
	string dir_path = "./image_9000/";
	string output_path = "./output/";
	Directory dir;
	vector<string> fileNames = dir.GetListFiles(dir_path, "*.jpg", false);

	for (int i = 0; i < fileNames.size(); i++)
	{
		string fileName = fileNames[i];
		string fileFullName = dir_path + fileName;
		string output1_path = output_path + fileName;
		cout << "file name:" << fileName << endl;
		cout << "file paht:" << fileFullName << endl << endl;

		//Image processing  
		Mat pScr;	
		pScr = imread(fileFullName, 1); 
		cout << "file channels:" << pScr.channels() << endl;
		if (pScr.channels() == 3)
			remove(fileFullName.c_str());
	}
	return 0;
}