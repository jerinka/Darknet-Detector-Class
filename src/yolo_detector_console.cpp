#include"yolo_detector.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "console_utils.h"
using namespace cv;
using namespace std;



int main2(int argc, char *argv[])
{
	auto start = std::chrono::steady_clock::now();
	std::vector<cv::Rect> roi_vec_person, roi_vec_face;

	String relativepath = "../../data/tiny_yolo_person/test";
	vector<String> filenames;
	cu::ConsoleUtils::getImages(relativepath, filenames);

	cout << "Number of images detected in exe path = "<< filenames.size() << endl;//to display no of files
	cv::Mat mat_img;

	cout << "Use webcam or Keep images in exe path, click enter to select next image" << endl<<endl;

	int type1, type2;
	cout << "Enter: 1- Webcam, 2- Image files " << endl;
	cin >> type1;
	cout << "Enter: 1- Tiny yolo face, 2- Tiny yolo person, 3- Yolov3" << endl;
	cin >> type2;
	std::shared_ptr<yoloDetector> yolodet(nullptr);
	if (type2 == 1)
		yolodet = make_shared<yoloDetector>("../../data/tiny_yolo_face/face.names", "../../data/tiny_yolo_face/yolov3_tiny_face.cfg", "../../data/tiny_yolo_face/yolov3tiny_face_71400.weights", 0.20);
	else if (type2 == 2)
		yolodet = make_shared<yoloDetector>("../../data/tiny_yolo_person/coco.names", "../../data/tiny_yolo_person/yolov3-tiny.cfg", "../../data/tiny_yolo_person/yolov3-tiny.weights", 0.20);
	else
		yolodet = make_shared<yoloDetector>("../../data/yolov3/coco.names", "../../data/yolov3/yolov3.cfg", "../../data/yolov3/yolov3.weights", 0.20);

	if (type1 == 2) {
		for (int i = 0; i < filenames.size(); ++i)
		{
			cout << filenames[i] << endl;
			mat_img = cv::imread(filenames[i], CV_LOAD_IMAGE_COLOR);

			if (!mat_img.data)                              // Check for invalid input
			{
				cout << "Could not open or find the image" << std::endl;
				return -1;
			}
			cv::imshow("window name", mat_img);

			yolodet->detect_yolo(mat_img, roi_vec_person);

			for (int i = 0; i < roi_vec_person.size(); i++) {
				rectangle(mat_img, cv::Rect(roi_vec_person[i].x, roi_vec_person[i].y, roi_vec_person[i].width, roi_vec_person[i].height), cv::Scalar(0, 0, 255), 2);
			}

			cv::imshow("window name", mat_img);



			auto end = std::chrono::steady_clock::now();
			std::chrono::duration<double> spent = end - start;
			start = end;
			std::cout << " Time: " << spent.count() << " sec \n";
			std::cout << " fps: " << 1 / spent.count() << " sec \n";

			waitKey(0);
		}
	}
	else
	{
		cv::VideoCapture cap(0); // open the default camera
		if (!cap.isOpened())  // check if we succeeded
			return -1;

		cv::Mat mat_img;
		for (int i = 0; i < 300; i++)
		{
			// image file
			//for(int j=0;j<15;j++)
			cap >> mat_img;
			yolodet->detect_yolo(mat_img, roi_vec_person);

			for (int i = 0; i < roi_vec_person.size(); i++) {

				rectangle(mat_img, cv::Rect(roi_vec_person[i].x, roi_vec_person[i].y, roi_vec_person[i].width, roi_vec_person[i].height), cv::Scalar(0, 0, 255), 2);
			}

			cv::imshow("window name", mat_img);

			auto end = std::chrono::steady_clock::now();
			std::chrono::duration<double> spent = end - start;
			start = end;
			std::cout << " Time: " << spent.count() << " sec \n";
			std::cout << " fps: " << 1 / spent.count() << " sec \n";

			if (cv::waitKey(30) == 27) break;
		}
	}

	return 0;
}


