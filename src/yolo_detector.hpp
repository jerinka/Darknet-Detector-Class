#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

//#ifdef _WIN32
//#define OPENCV
//#define GPU
//#endif


#include "yolo_v2_class.hpp"	// imported functions from DLL

#include <opencv2/opencv.hpp>			// C++
#include "opencv2/core/version.hpp"

#include "opencv2/videoio/videoio.hpp"
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")


class yoloDetector
{

	std::string  names_file;// = "data/tiny_yolo_basic/coco.names";
	std::string  cfg_file;// = "data/tiny_yolo_basic/yolov3-tiny.cfg";
	std::string  weights_file;// = "data/tiny_yolo_basic/yolov3-tiny.weights";
	std::string filename;// = "dog.jpg"; // "F:\\ObjectDetection\\darknet\\darknet_console\\scripts\\outpy.avi";
	int gpuid = 0;
	float const thresh = 0.20;

	std::vector<std::string> obj_names;

public:

	Detector detector = Detector(cfg_file, weights_file, gpuid);

	yoloDetector(std::string  names_file1, std::string  cfg_file1, std::string  weights_file1, float thresh1) :names_file(names_file1), weights_file(weights_file1), cfg_file(cfg_file1), thresh(thresh1)
	{
		obj_names = objects_names_from_file(names_file1);
	}

	~yoloDetector()
	{
		std::cout << "yoloDetector destructor" << std::endl;
	}

	std::vector<std::string> objects_names_from_file(std::string const filename) {
		//std::cout<<__FUNCTION__<<"\n";
		std::ifstream file(filename);
		std::vector<std::string> file_lines;
		if (!file.is_open()) return file_lines;
		for (std::string line; getline(file, line);) file_lines.push_back(line);
		std::cout << "object names loaded \n";
		return file_lines;
	}

	void get_roi_vec(cv::Mat mat_img, std::vector<cv::Rect> &roi_vec, std::vector<bbox_t> result_vec) {
		roi_vec.clear();
		int width = mat_img.size().width;
		int height = mat_img.size().height;
		for (auto &i : result_vec) {
			//cv::Scalar color = obj_id_to_color(i.obj_id);
			//cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
			//cv::imshow("test1",mat_img)
			if (obj_names.size() > i.obj_id) {
				
				std::string obj_name = obj_names[i.obj_id];
				if (obj_name == "person" || obj_name == "face") {
					
					cv::Rect roi = cv::Rect(i.x, i.y, i.w, i.h);
					if (roi.x < 0)
						roi.x = 0;
					if (roi.y < 0)
						roi.y = 0;

					if (roi.x + roi.width > width)
						roi.width = width - roi.x;
					if (roi.y + roi.height > height)
						roi.height = height - roi.y;
					roi_vec.push_back(roi);
				}
			}
		}
	}

	void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec)
	{
		//std::cout<<__FUNCTION__<<"\n";
		int current_det_fps = -1;
		int current_cap_fps = -1;

		for (auto &i : result_vec) {
			cv::Scalar color = obj_id_to_color(i.obj_id);
			cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
			if (obj_names.size() > i.obj_id) {
				std::string obj_name = obj_names[i.obj_id];
				if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
				cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
				int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
				cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 30, 0)),
					cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
					color, CV_FILLED, 8, 0);
				putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			}
		}
		if (current_det_fps >= 0 && current_cap_fps >= 0) {
			std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
			putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
		}
	}


	void detect_yolo(cv::Mat mat_img, std::vector<cv::Rect> &roi_vec)
	{
		//std::cout<<__FUNCTION__<<"\n";
		//cv::Mat mat_img(data_ptr->h,data_ptr->w,CV_32F, data_ptr->data);
		//Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP);
		std::vector<bbox_t> result_vec = detector.detect(mat_img, thresh);

		get_roi_vec(mat_img, roi_vec, result_vec);
		//draw_boxes(mat_img, result_vec);

		//return mat_img;
	}

};


