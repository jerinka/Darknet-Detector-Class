//#include <iostream>
//#include <iomanip> 
//#include <string>
//#include <vector>
//#include <queue>
//#include <fstream>
//#include <thread>
//#include <atomic>
//#include <mutex>              // std::mutex, std::unique_lock
//#include <condition_variable> // std::condition_variable
//
//#ifdef _WIN32
//#define OPENCV
//#define GPU
//#endif
//
//
//#include "yolo_v2_class.hpp"	// imported functions from DLL
//
//#include <opencv2/opencv.hpp>			// C++
//#include "opencv2/core/version.hpp"
//
//#include "opencv2/videoio/videoio.hpp"
//#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
//#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
//
//void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
//	int current_det_fps = -1, int current_cap_fps = -1)
//{
//	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
//
//	for (auto &i : result_vec) {
//
//		
//		if (obj_names.size() > i.obj_id) {
//			std::string obj_name = obj_names[i.obj_id];
//			if (obj_name == "dog") {
//				cv::Scalar color = obj_id_to_color(i.obj_id);
//				cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
//
//				if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
//				cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
//				int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
//				cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 30, 0)),
//					cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
//					color, CV_FILLED, 8, 0);
//				putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
//			}
//		}
//	}
//	if (current_det_fps >= 0 && current_cap_fps >= 0) {
//		std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
//		putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
//	}
//}
//
//
//void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
//	for (auto &i : result_vec) {
//		if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
//		std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
//			<< ", w = " << i.w << ", h = " << i.h
//			<< std::setprecision(3) << ", prob = " << i.prob << std::endl;
//	}
//}
//
//std::vector<std::string> objects_names_from_file(std::string const filename) {
//	std::ifstream file(filename);
//	std::vector<std::string> file_lines;
//	if (!file.is_open()) return file_lines;
//	for (std::string line; getline(file, line);) file_lines.push_back(line);
//	std::cout << "object names loaded \n";
//	return file_lines;
//}
//
//
//int main5(int argc, char *argv[])
//{
//	std::string  names_file = "data/coco.names";
//	std::string  cfg_file = "cfg/yolov3-tiny.cfg";
//	std::string  weights_file = "weights/yolov3-tiny.weights";
//	std::string filename = "dog.jpg"; // "F:\\ObjectDetection\\darknet\\darknet_console\\scripts\\outpy.avi";
//
//
//	float const thresh = (argc > 5) ? std::stof(argv[5]) : 0.20;
//
//	Detector detector(cfg_file, weights_file);
//
//	auto obj_names = objects_names_from_file(names_file);
//
//	while (true)
//	{
//
//		   	// image file
//				cv::Mat mat_img = cv::imread(filename);
//
//				auto start = std::chrono::steady_clock::now();
//				std::vector<bbox_t> result_vec = detector.detect(mat_img);
//				auto end = std::chrono::steady_clock::now();
//				std::chrono::duration<double> spent = end - start;
//				std::cout << " Time: " << spent.count() << " sec \n";
//
//				//result_vec = detector.tracking_id(result_vec);	// comment it - if track_id is not required
//				draw_boxes(mat_img, result_vec, obj_names);
//
//				cv::imshow("window name", mat_img);
//				//show_console_result(result_vec, obj_names);
//				cv::waitKey(1);
//
//	}
//
//	return 0;
//}