#include "yolo_person_face.h"
#include <chrono>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <glob.h>
//#include <iostream>
//#include <string>

int img_x = 0;
int img_y = 0;
int img_width = 480;   
int img_height = 480;
//#include "resnet_ssd_face.hpp"

using namespace cv;
using namespace std;

std::string save_file_name = "img_";
std::string save_dest = "dataset/";
int idx = 0;

void SaveDataset(Mat frame, std::vector<cv::Rect> roi_vec) {

	if (roi_vec.size() > 0) {
		string image_name;
		image_name.append(save_dest);
		image_name.append(save_file_name);
		image_name.append(std::to_string(idx));
		image_name.append(".jpg");

		string txt_name;
		txt_name.append(save_dest);
		txt_name.append(save_file_name);
		txt_name.append(std::to_string(idx));
		txt_name.append(".txt");

		ofstream myfile;
		myfile.open(txt_name);

		cout << "image_name" << image_name << std::endl;
		cout << "txt_name : " << txt_name << std::endl;

		cv::imwrite(image_name, frame);

		for (int i = 0; i < roi_vec.size(); i++) {
			int x, y, w, h;
			int frame_width, frame_height;

			float new_x, new_y, new_w, new_h;

			frame_width = frame.size().width;
			frame_height = frame.size().height;
			x = roi_vec[i].x;
			y = roi_vec[i].y;
			w = roi_vec[i].width;
			h = roi_vec[i].height;

			new_x = float(x) + float(w / 2);
			new_y = float(y) + float(h / 2);

			new_x = new_x / frame_width;
			new_y = new_y / frame_height;

			new_w = float(w) / frame_width;
			new_h = float(h) / frame_height;

			myfile << 0;
			myfile << " ";
			myfile << new_x;
			myfile << " ";
			myfile << new_y;
			myfile << " ";

			myfile << new_w;
			myfile << " ";

			myfile << new_h;
			myfile << "\n";

			std::cout << "new_x : " << new_x << "new_y : " << new_y << " new_w : " << new_w << "new_h: " << new_h << std::endl;
		}
		myfile.close();
		idx++;
	}
}

#if LIVE_CAMERA
int main() {
	VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera" << endl;
	}

	YOLO_PERSON_FACE yolo_person_face;
	Mat cameraFrame, cameraFrame_clone;
	Mat mat_crop, mat_crop_clone;
	Rect myROI(img_x, img_y, img_width, img_height);

	std::vector<cv::Rect> roi_vec_person, roi_vec_face;
	auto start = std::chrono::steady_clock::now();

	std::vector<roi_data> person_info;
	std::vector<roi_data> person_info_vec;

	//unconditional loop
	while (true) {

		stream1.read(cameraFrame);

		cameraFrame_clone = cameraFrame.clone();

		cout << "########## size111 : #####" << cameraFrame.size().width << " x " << cameraFrame.size().height;

		mat_crop = cameraFrame.clone();

		if (img_width != 0 && img_height != 0)
			mat_crop = mat_crop(myROI);

		mat_crop_clone = mat_crop.clone();

		//yolo_detect_person.detect_yolo(mat_crop,roi_vec_person);
		yolo_person_face.get_person_face_rect(mat_crop, person_info);

		cout << " size of person_info : " << person_info.size() << endl;

		if (person_info.size() > 0) {
			cout << " size of person facde : " << person_info[0].face_rect.width << endl;
		}

		for (int i = 0; i < person_info.size(); i++) {
			rectangle(mat_crop, cv::Rect(person_info[i].person_rect.x, person_info[i].person_rect.y, person_info[i].person_rect.width, person_info[i].person_rect.height), Scalar(0, 0, 255), 2);
			rectangle(mat_crop, cv::Rect(person_info[i].face_rect.x, person_info[i].face_rect.y, person_info[i].face_rect.width, person_info[i].face_rect.height), Scalar(255, 0, 0), 2);
		}

		imshow("LIVE", mat_crop);

		if (waitKey(30) >= 0)
			break;

		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> spent = end - start;
		start = end;
		std::cout << " Time: " << spent.count() << " sec \n";
		std::cout << " fps: " << 1 / spent.count() << " frames \n";
	}
	return 0;
}
#else

int main(){
	YOLO_PERSON_FACE yolo_person_face;
	Mat cameraFrame, cameraFrame_clone;
	Mat mat_crop, mat_crop_clone;

	std::vector<cv::Rect> roi_vec_person, roi_vec_face;
	auto start = std::chrono::steady_clock::now();

	std::vector<roi_data> person_info;

	String folder = "F:/Darknet/darknet_person_face_classifier_3/build/darknet/dataset/person_face/*.jpg";
	vector<String> filenames;
	glob(folder, filenames);
	std::sort(filenames.begin(), filenames.end());
	
	cout << filenames.size() << endl;//to display no of files

	for (int k = 0; k < 101; k++)
	{
		//YOLO_PERSON_FACE yolo_person_face;
		for (int i = 0; i < filenames.size(); ++i)
		{


			cout << filenames[i] << endl;
			cameraFrame = cv::imread(filenames[i], CV_LOAD_IMAGE_COLOR);

			if (!cameraFrame.data)                              // Check for invalid input
			{
				cout << "Could not open or find the image" << std::endl;
				return -1;
			}

			mat_crop = cameraFrame.clone();

			mat_crop_clone = mat_crop.clone();


			//YOLO_PERSON_FACE yolo_person_face;
			yolo_person_face.get_person_face_rect(mat_crop, person_info);

			cout << " size of person_info : " << person_info.size() << endl;

			if (person_info.size() > 0) {
				cout << " size of person facde : " << person_info[0].face_rect.width << endl;
			}

			for (int i = 0; i < person_info.size(); i++) {
				cout << "Classification : " << person_info[0].classification_result << endl;
				rectangle(mat_crop, cv::Rect(person_info[i].person_rect.x, person_info[i].person_rect.y, person_info[i].person_rect.width, person_info[i].person_rect.height), Scalar(0, 0, 255), 2);
				rectangle(mat_crop, cv::Rect(person_info[i].face_rect.x, person_info[i].face_rect.y, person_info[i].face_rect.width, person_info[i].face_rect.height), Scalar(255, 0, 0), 2);
			}

			imshow("LIVE", mat_crop);

			if (waitKey(2) >= 0)
				break;

			auto end = std::chrono::steady_clock::now();
			std::chrono::duration<double> spent = end - start;
			start = end;
			std::cout << " Time: " << spent.count() << " sec \n";
			std::cout << " fps: " << 1 / spent.count() << " frames \n";
			person_info.clear();
		}
	}
 	return 0;


}

#endif
