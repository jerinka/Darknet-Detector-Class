#include "yolo_person_face.h"
#include <chrono>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "console_utils.h"

int img_x = 0;
int img_y = 0;
int img_width = 480;   
int img_height = 480;

using namespace cv;
using namespace std;

std::string save_file_name = "img_";
std::string save_dest = "dataset/";
int idx = 0;


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

	String relativepath = "../../data/tiny_yolo_person/test";
	vector<String> filenames;
	cu::ConsoleUtils::getImages(relativepath, filenames);
	
	cout << filenames.size() << endl;//to display no of files

	int on = 1;
	while(on)
	{
		//YOLO_PERSON_FACE yolo_person_face;
		for (int i = 0; i < filenames.size(); ++i)
		{


			//cout << filenames[i] << endl;
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

			//cout << " size of person_info : " << person_info.size() << endl;

			//if (person_info.size() > 0) {
			//	cout << " size of person facde : " << person_info[0].face_rect.width << endl;
			//}

			for (int i = 0; i < person_info.size(); i++) {
				cout << "Classification : " << person_info[0].classification_result << endl;
				rectangle(mat_crop, cv::Rect(person_info[i].person_rect.x, person_info[i].person_rect.y, person_info[i].person_rect.width, person_info[i].person_rect.height), Scalar(0, 0, 255), 2);
				rectangle(mat_crop, cv::Rect(person_info[i].face_rect.x, person_info[i].face_rect.y, person_info[i].face_rect.width, person_info[i].face_rect.height), Scalar(255, 0, 0), 2);

				int index = person_info[i].classification_result;
				char txt[50];
				sprintf_s(txt, "C= %d", index);
				//cout << txt << endl;
				cv::putText(mat_crop,
					txt,
					cv::Point(person_info[i].face_rect.x, person_info[i].face_rect.y), // Coordinates
					cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
					1, // Scale. 2.0 = 2x bigger
					cv::Scalar(0, 0, 255), // BGR Color
					1, // Line Thickness (Optional)
					CV_AA); // Anti-alias (Optional)

				sprintf_s(txt, "C= %d", person_info[i].person_id);
				cv::putText(mat_crop,
					txt,
					cv::Point(person_info[i].face_rect.x, person_info[i].face_rect.br().y), // Coordinates
					cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
					1, // Scale. 2.0 = 2x bigger
					cv::Scalar(0, 0, 255), // BGR Color
					1, // Line Thickness (Optional)
					CV_AA); // Anti-alias (Optional)

			}

			imshow("LIVE", mat_crop);

			auto end = std::chrono::steady_clock::now();
			std::chrono::duration<double> spent = end - start;
			start = end;
			std::cout << " Time: " << spent.count() << " sec \n";
			std::cout << " fps: " << 1 / spent.count() << " frames \n";
			person_info.clear();

			if ((char)27 == (char)waitKey(30)) { on = 0; break; }
		}

	}
 	return 0;


}

#endif
