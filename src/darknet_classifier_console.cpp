//#define _CRTDBG_MAP_ALLOC
//#include <cstdlib>
//#include <crtdbg.h>

#include "darknet_classifier.hpp"
#include <chrono>
#include <iostream>
#include "console_utils.h"

using namespace cv;
using namespace std;


int main21(int argc, char *argv[])
{
	auto start = std::chrono::steady_clock::now();
	
	String relativepath = "../../data/capmask/test";
	vector<String> filenames;
	cu::ConsoleUtils::getImages(relativepath, filenames);

	cout << filenames.size() << endl;//to display no of files
	cv::Mat mat_img;

	Darknet_Classifier darknet_class("../../data/capmask/capmask.data", "../../data/capmask/capmask.cfg", "../../data/capmask/capmask_75000.weights");

	int on = 1;
	while(on)
	for (int i = 0; i < filenames.size(); ++i)
	{
		//Darknet_Classifier darknet_class("data/capmask/capmask.data", "data/capmask/capmask.cfg", "data/capmask/capmask_2000.weights");

		//cout << filenames[i] << endl;
		mat_img = cv::imread(filenames[i], CV_LOAD_IMAGE_COLOR);

		if (!mat_img.data)                              // Check for invalid input
		{
			cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		
		int index = darknet_class.darknet_classifier(mat_img);
		char txt[50];
		sprintf_s(txt, "C= %d",index);
		//cout << txt << endl;
		cv::putText(mat_img,
			txt,
			cv::Point(10, 20), // Coordinates
			cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
			1, // Scale. 2.0 = 2x bigger
			cv::Scalar(0, 0, 255), // BGR Color
			1, // Line Thickness (Optional)
			CV_AA); // Anti-alias (Optional)

		cv::imshow("window name", mat_img);

		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> spent = end - start;
		start = end;
		//std::cout << " Time: " << spent.count() << " sec \n";
		//std::cout << " fps: " << 1 / spent.count() << " sec \n";

		if ((char)27 == (char)waitKey(30)) { on = 0; break; }
	}
	return 0;
}