#pragma once
#ifndef CONSOLE_UTILS_H
#define CONSOLE_UTILS_H

#include <opencv2/opencv.hpp>
#include<opencv2/core/utils/filesystem.hpp>

namespace cu
{
	class ConsoleUtils
	{
	public:
	static void getImages(cv::String relativepath, std::vector<cv::String>& filenames)
	{
		/*Inputs:
		relativepath : eg,  "../../data/capmask/test", wrt current working directory
		Outputs:
		filenames    : vector of full path names 
		*/
		cv::String basepath = cv::utils::fs::getcwd();
		cv::String Folder = basepath + "/" + relativepath;

		std::vector<cv::String> filenames1, filenames2;
		cv::String folder1 = Folder + "/*.png";
		cv::String folder2 = Folder + "/*.jpg";
		cv::glob(folder1, filenames1);
		cv::glob(folder2, filenames2);
		filenames1.insert(filenames1.end(), filenames2.begin(), filenames2.end());
		filenames = filenames1;

		std::sort(filenames.begin(), filenames.end());
	}

	//static void SaveDataset(Mat frame, std::vector<cv::Rect> roi_vec) {

	//	if (roi_vec.size() > 0) {
	//		string image_name;
	//		image_name.append(save_dest);
	//		image_name.append(save_file_name);
	//		image_name.append(std::to_string(idx));
	//		image_name.append(".jpg");

	//		string txt_name;
	//		txt_name.append(save_dest);
	//		txt_name.append(save_file_name);
	//		txt_name.append(std::to_string(idx));
	//		txt_name.append(".txt");

	//		ofstream myfile;
	//		myfile.open(txt_name);

	//		cout << "image_name" << image_name << std::endl;
	//		cout << "txt_name : " << txt_name << std::endl;

	//		cv::imwrite(image_name, frame);

	//		for (int i = 0; i < roi_vec.size(); i++) {
	//			int x, y, w, h;
	//			int frame_width, frame_height;

	//			float new_x, new_y, new_w, new_h;

	//			frame_width = frame.size().width;
	//			frame_height = frame.size().height;
	//			x = roi_vec[i].x;
	//			y = roi_vec[i].y;
	//			w = roi_vec[i].width;
	//			h = roi_vec[i].height;

	//			new_x = float(x) + float(w / 2);
	//			new_y = float(y) + float(h / 2);

	//			new_x = new_x / frame_width;
	//			new_y = new_y / frame_height;

	//			new_w = float(w) / frame_width;
	//			new_h = float(h) / frame_height;

	//			myfile << 0;
	//			myfile << " ";
	//			myfile << new_x;
	//			myfile << " ";
	//			myfile << new_y;
	//			myfile << " ";

	//			myfile << new_w;
	//			myfile << " ";

	//			myfile << new_h;
	//			myfile << "\n";

	//			std::cout << "new_x : " << new_x << "new_y : " << new_y << " new_w : " << new_w << "new_h: " << new_h << std::endl;
	//		}
	//		myfile.close();
	//		idx++;
	//	}
	//}


	};





}
#endif // !CONSOLE_UTILS_H

