//#define _CRTDBG_MAP_ALLOC
//#include <cstdlib>
//#include <crtdbg.h>

#include "darknet_classifier.hpp"
#include        <chrono>
#include<iostream>

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


int main7(int argc, char *argv[])
{
	auto start = std::chrono::steady_clock::now();
	
	String folder = "F:/Darknet/darknet_person_face_classifier_3/build/darknet/dataset/person_face/*.jpg";
	vector<String> filenames;
	glob(folder, filenames);
	std::sort(filenames.begin(), filenames.end());

	cout << filenames.size() << endl;//to display no of files
	cv::Mat mat_img;

	Darknet_Classifier darknet_class("data/capmask/capmask.data", "data/capmask/capmask.cfg", "data/capmask/capmask_2000.weights");

	for(int k=0;k<10;k++)
	for (int i = 0; i < filenames.size(); ++i)
	{

		//Darknet_Classifier darknet_class("data/capmask/capmask.data", "data/capmask/capmask.cfg", "data/capmask/capmask_2000.weights");

		cout << filenames[i] << endl;
		mat_img = cv::imread(filenames[i], CV_LOAD_IMAGE_COLOR);

		if (!mat_img.data)                              // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		cv::imshow("window name", mat_img);
		cv::waitKey(10);

		int index = darknet_class.darknet_classifier(mat_img);
		std::cout << " index " << index << std::endl;

		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> spent = end - start;
		start = end;
		std::cout << " Time: " << spent.count() << " sec \n";
		std::cout << " fps: " << 1 / spent.count() << " sec \n";

		cv::waitKey(30);
	}

	return 0;
}