#ifndef YOLO_PERSON_FACE_H
#define YOLO_PERSON_FACE_H


#include "yolo_detector.hpp"
#include "darknet_classifier.hpp"

#define YOLO_DEBUG 1
#define TEST_YOLO 1

using namespace cv;
using namespace std;

struct roi_flag{
    cv::Rect rect;
    int dist;
    bool flag;
    bool occupied_flag;
};

struct roi_data{
    cv::Rect face_rect,person_rect;
    int person_id;
    int classification_result;
};

struct person_track{
    cv::Rect face_rect,person_rect;
    int person_id;
    std::vector<int> classification_result_vector;

    cv::Point cent;
    int face_count;
    bool frame_flag;
};

struct gperson_track_dist_id{
    int eclu_distance;
    int gperson_id;
};

class YOLO_PERSON_FACE{

private :

public :

    std::vector<person_track> g_person_track_info;
    int PAST_CNT = 5;
    int FACE_CNT = 2;

    std::vector<roi_data> track_result_info;

    //        int eclu_dist = 0;
    //        int eclu_dist_thresh = 200;				//CHANGE RESPECTIVE TO FRAME SIZE / FACTOR  09/08/2018

    //        double diff_x; //calculating number to square in next step
    //       double diff_y;

    int person_id = 0;

    //       double dist;
    //       double min_dist;// = frame.size().width;

    yoloDetector yolo_detect_person = yoloDetector("data/tiny_yolo_person/coco.names", "data/tiny_yolo_person/yolov3-tiny.cfg", "data/tiny_yolo_person/yolov3-tiny.weights", 0.20);
	yoloDetector yolo_detect_face = yoloDetector("data/tiny_yolo_face/face.names", "data/tiny_yolo_face/yolov3-tiny.cfg", "data/tiny_yolo_face/yolov3-tiny_71400.weights", 0.20);
    Darknet_Classifier darknet_class = Darknet_Classifier("data/capmask/capmask.data", "data/capmask/capmask.cfg", "data/capmask/capmask_2000.weights");


    YOLO_PERSON_FACE()
	{
    }

    ~YOLO_PERSON_FACE(){
		std::cout << "yolo person face classifier destructor" << std::endl;
    }

    void draw_rect(cv::Mat &frame,std::vector<cv::Rect> roi_vec,Scalar color=Scalar(0,0,255)){
        for(int i = 0; i < roi_vec.size() ; i++){

            rectangle(frame, cv::Rect(roi_vec[i].x, roi_vec[i].y,roi_vec[i].width, roi_vec[i].height), color, 2);

        }
    }

    void person_detect(cv::Mat& frame, std::vector<cv::Rect>& roi_person){
        yolo_detect_person.detect_yolo(frame,roi_person);
    }



     void face_detect(cv::Mat& frame, std::vector<cv::Rect>& roi_face){
        yolo_detect_face.detect_yolo(frame,roi_face);
    }


    void  map_face_to_person (cv::Mat& frame,std::vector<cv::Rect>& roi_person,std::vector<cv::Rect>& roi_face, std::vector<roi_data>& person_info_vec){

		person_info_vec.clear();
        std::vector<roi_flag> roi_person_vec, roi_face_vec;

        for(int i = 0; i < roi_person.size(); i++ ){

            roi_flag temp_roi_person;
            temp_roi_person.rect = roi_person[i];
            temp_roi_person.dist = 100000;
            temp_roi_person.occupied_flag = false;
            temp_roi_person.flag = false;
            roi_person_vec.push_back(temp_roi_person);
        }

        for(int i = 0; i < roi_face.size(); i++ ){

            roi_flag temp_roi_face;
            temp_roi_face.rect = roi_face[i];
            temp_roi_face.dist = 100000;
            temp_roi_face.flag = false;
            temp_roi_face.occupied_flag = false;
            roi_face_vec.push_back(temp_roi_face);

        }



        for(int i = 0;i < roi_person_vec.size() ; i++){

            for(int j = 0 ; j < roi_face_vec.size() ; j++){
                roi_face_vec[j].flag = false;
            }


            for(int j = 0 ; j < roi_face_vec.size() ; j++){

                if(roi_face_vec[j].occupied_flag == false){

                    if(roi_face_vec[j].rect.x + roi_face_vec[j].rect.width / 2 > roi_person_vec[i].rect.x && roi_face_vec[j].rect.x + roi_face_vec[j].rect.width / 2  < roi_person_vec[i].rect.br().x){

                        if(roi_face_vec[j].rect.br().y > roi_person_vec[i].rect.tl().y && (roi_face_vec[j].rect.br().y-roi_face_vec[j].rect.width/4) < roi_person_vec[i].rect.y + roi_person_vec[i].rect.height+0){
                            roi_face_vec[j].flag = true;
                            cv::Point face_pt,person_pt;
                            double diff_x; //calculating number to square in next step
                            double diff_y;

                            double dist;
                            // = frame.size().width;

                            person_pt.x = (roi_person_vec[i].rect.x + roi_person_vec[i].rect.width) / 2;
                            person_pt.y = (roi_person_vec[i].rect.y + roi_person_vec[i].rect.height) / 5;

                            face_pt.x = (roi_face_vec[j].rect.x + roi_face_vec[j].rect.width) / 2;
                            face_pt.y = (roi_face_vec[j].rect.y + roi_face_vec[j].rect.height) / 5;

                            diff_x = person_pt.x - face_pt.x;
                            diff_y = person_pt.y - face_pt.y;

                            dist = pow(diff_x, 2) + pow(diff_y, 2);
                            dist = sqrt(dist);

                            roi_face_vec[j].dist = dist;


                        }

                    }
                }
            }

            int index = -1;
            double min_dist = frame.size().width;
            for(int j = 0 ; j < roi_face_vec.size() ; j++){
                if(roi_face_vec[j].occupied_flag == false){
                    if(roi_face_vec[j].flag == true){
                        if(roi_face_vec[j].dist < min_dist){
                            min_dist = roi_face_vec[j].dist;
                            index = j;
                        }

                    }
                }
            }
            roi_data temp_person;
            temp_person.person_rect = roi_person_vec[i].rect;
            if(index != -1){
                temp_person.face_rect = roi_face_vec[index].rect;
                roi_face_vec[index].occupied_flag = true;
            }
            else{
                temp_person.face_rect  = cv::Rect(0,0,0,0);
            }

            person_info_vec.push_back(temp_person);


        }
        for(int j = 0 ; j < roi_face_vec.size() ; j++){
            if(roi_face_vec[j].occupied_flag == false){
                roi_data temp_person;
                temp_person.person_rect = Rect(0,0,0,0);
                temp_person.face_rect = roi_face_vec[j].rect;
                roi_face_vec[j].occupied_flag = true;

                person_info_vec.push_back(temp_person);
            }
        }

		roi_person_vec.clear();
		roi_face_vec.clear();
    }

     void get_person_face_rect(cv::Mat &frame, std::vector<roi_data>& person_info_vec){

		 person_info_vec.clear();
		std::vector<cv::Rect> roi_vec_person;
		std::vector<cv::Rect> roi_vec_face;

        person_detect(frame, roi_vec_person);
        face_detect(frame, roi_vec_face);

		std::cout << "size of person :" << roi_vec_person.size() << "size of face : " << roi_vec_face.size() <<endl;

        map_face_to_person(frame,roi_vec_person,roi_vec_face, person_info_vec);

        person_info_vec = yolo_darknet_classifier(person_info_vec,frame.clone());
        //person_info_vec = yolo_track(frame,person_info_vec);
        person_info_vec = yolo_track_2(frame,person_info_vec);

        cout << "size of person info vec : " << person_info_vec.size() <<endl;
    }




#if 0
    std::vector<roi_data> get_person_face_rect(cv::Mat &frame){


         person_detect(frame, roi_vec_person);
        //yolo_detect_person.detect_yolo(frame,roi_vec_person);
        std::vector<roi_data> person_info_vec;


        face_detect(frame, roi_vec_face);
        //yolo_detect_face.detect_yolo(frame,roi_vec_face);


        if(roi_vec_face.size() > 0){

            for(int i = 0; i < roi_vec_face.size() ; i++){

                //rectangle(frame, cv::Rect(roi_vec_face[i].x, roi_vec_face[i].y,roi_vec_face[i].width, roi_vec_face[i].height), Scalar(0,0,255), 2);

            }
        }


        std::vector<double> dist_vec;
        cv::Rect roi_person,roi_face;
        cv::Point face_pt,person_pt;

        int person_id = 0;

        double diff_x; //calculating number to square in next step
        double diff_y;

        double dist;
        double min_dist;// = frame.size().width;


        if(roi_vec_person.size() > 0){

            for(int i = 0; i < roi_vec_person.size() ; i++){
                roi_person = roi_vec_person[i];
                person_pt.x = (roi_person.x + roi_person.width) / 2;
                person_pt.y = (roi_person.y + roi_person.height) / 5;
                if(roi_vec_face.size() > 0){
                    dist_vec.clear();
                    for(int j = 0; j < roi_vec_face.size() ; j++){
                        roi_face = roi_vec_face[j];
                        face_pt.x = (roi_face.x + roi_face.width) / 2;
                        face_pt.y = (roi_face.y + roi_face.height) / 5;

                        diff_x = person_pt.x - face_pt.x;

                        diff_y = person_pt.y - face_pt.y;


                        dist = pow(diff_x, 2) + pow(diff_y, 2);
                        dist = sqrt(dist);

                        //cout << "dist : " << dist <<endl;
                        dist_vec.push_back(dist);


                    }
                    //cout << " size of dist_vec " << dist_vec.size() << endl;
                    min_dist = frame.size().width;
                    int index = -1;
                    for(int j = 0; j < dist_vec.size() ; j++){
                        if(dist_vec[j] < min_dist){
                            min_dist = dist_vec[j];
                            index = j;
                        }
                    }

                    //cout << "min_dist " << min_dist << "index : " << index  << endl;
                    roi_face = roi_vec_face[index];


                    //cout << "#########" << roi_face.x + roi_face.width / 2 << " , " << roi_person.x << " , " <<  roi_face.x+roi_face.width /2  << " , " << roi_person.x + roi_person.width << " , " <<  roi_face.y + roi_face.height /2 << " , " <<  roi_person.y << " , " <<  roi_face.y + roi_face.height /2 << " , " <<  roi_person.y + roi_person.height <<endl;




                    if(roi_face.x + roi_face.width / 2 > roi_person.x && roi_face.x+roi_face.width /2  < roi_person.x + roi_person.width && roi_face.y + roi_face.height /2 > roi_person.y && roi_face.y + roi_face.height /2 < roi_person.y + roi_person.height){

                        roi_data person_info;
                        person_info.person_rect =roi_person;
                        person_info.face_rect = roi_face;
                        person_info.person_id = person_id;
                        person_id ++;
                        person_info.classification_result = 0;

                        cout << "person_id " << person_id <<endl;
                        person_info_vec.push_back(person_info);
                        roi_vec_face.erase(roi_vec_face.begin() + index);

                    }


                }
                else{
                    roi_data person_info;
                    person_info.person_rect =roi_person;
                    person_info.face_rect.x = 0;
                    person_info.face_rect.y = 0;
                    person_info.face_rect.width = 0;
                    person_info.face_rect.height = 0;
                    person_info.person_id = person_id;
                    person_id ++;
                    person_info.classification_result = 0;

                    person_info_vec.push_back(person_info);
                }

            }


        }


    }


    ////ADDED 25/07/2018 /////

    if (person_info_vec.size() > 0){

        person_info_vec = yolo_darknet_classifier(person_info_vec,frame.clone());
        person_info_vec = yolo_track(frame,person_info_vec);

    }

    /////////////////////////
    return person_info_vec;
}

#endif

///////////////// 10/08/2018 //////////////////////////////

void push_person_gtrack(roi_data person_info){
    person_track temp_person_track;

    temp_person_track.face_count = FACE_CNT;
    temp_person_track.cent.x     = person_info.face_rect.x + person_info.face_rect.width / 2;
    temp_person_track.cent.y     = person_info.face_rect.y + person_info.face_rect.height / 2;
    temp_person_track.face_rect = person_info.face_rect;
    temp_person_track.person_rect = person_info.person_rect;
    temp_person_track.person_id = person_id;
    person_id++;

    temp_person_track.frame_flag = true;
    temp_person_track.classification_result_vector.push_back(person_info.classification_result);
    g_person_track_info.push_back(temp_person_track);
}

/*
 *
 * 	cv::Rect face_rect,person_rect;
        int person_id;
        int classification_result;
 */

void push_person_track_result(roi_data person_info){

    roi_data temp_person;
    temp_person.face_rect   = person_info.face_rect;
    temp_person.person_rect = person_info.person_rect;
    temp_person.person_id   = person_info.person_id;

    temp_person.classification_result = person_info.classification_result;
    track_result_info.push_back(temp_person);


}





void decrement_update_g_tracklist(){
#if 1

    for (int j = 0; j < g_person_track_info.size(); j++){

        g_person_track_info[j].frame_flag = false;
        g_person_track_info[j].face_count --;

    }

    // TO REMOVE STRUCTURE IF FRAME


    if(g_person_track_info.size() > 0){
        std::vector<person_track> temp_g_person_track_info;

		std::cout << "size 2 of g_person_track_info " << g_person_track_info.size() << endl;
        for (int j = 0; j < g_person_track_info.size(); j++){

            if(g_person_track_info[j].face_count > 0){
                temp_g_person_track_info.push_back(g_person_track_info[j]);
            }
        }
		g_person_track_info.clear();
        g_person_track_info = temp_g_person_track_info;
    }
#endif
}


std::vector<roi_data> yolo_track_2 (Mat frame, std::vector<roi_data> person_info){



    track_result_info.clear();

    decrement_update_g_tracklist();				// DECREMENTS AND REMOVES GLOBAL PERSON TRACK INFO VECTOR


//	std::cout << "#################### person_info size ####################" <<person_info.size() << endl;


    if(person_info.size() > 0){

        for(int i = 0 ; i < person_info.size(); i++){

            if((person_info[i].face_rect.x != 0 || person_info[i].face_rect.y != 0 || person_info[i].face_rect.width != 0 || person_info[i].face_rect.height != 0) ){

                if(g_person_track_info.size() > 0){

		//			std::cout << "#################### g_person_track_info.size() ####################" <<g_person_track_info.size() << endl;

                    std::vector<gperson_track_dist_id> track_dist_id;
                    int temp_dist = person_info[i].face_rect.width * 3;//eclu_dist_thresh;

                    for (int j = 0; j < g_person_track_info.size() ; j++){

                        int eclu_dist = 0;
                        //int eclu_dist_thresh = 200;				//CHANGE RESPECTIVE TO FRAME SIZE / FACTOR  09/08/2018

                        double diff_x; //calculating number to square in next step
                        double diff_y;
                        if(g_person_track_info[j].frame_flag == false){

                            diff_x = g_person_track_info[j].cent.x - (person_info[i].face_rect.x + person_info[i].face_rect.width / 2);
                            diff_y = g_person_track_info[j].cent.y - (person_info[i].face_rect.y + person_info[i].face_rect.height / 2);

                            eclu_dist = pow(diff_x, 2) + pow(diff_y, 2);
                            eclu_dist = sqrt(eclu_dist);

                            if (temp_dist > eclu_dist){
                                gperson_track_dist_id temp_track_dist_id;
                                temp_track_dist_id.eclu_distance = eclu_dist;
                                temp_track_dist_id.gperson_id = j;
                                track_dist_id.push_back(temp_track_dist_id);

                            }
                        }
                    }

                    int min_index = -1;
                    if(track_dist_id.size() > 0){
                        sort(track_dist_id.begin(), track_dist_id.end(), [](const gperson_track_dist_id a, const gperson_track_dist_id b) {return a.eclu_distance < b.eclu_distance; });
                        //min_index = track_dist_id[0].gperson_id;

                        for (int m = 0;  m < track_dist_id.size(); m++){
                //            cout << "track id" << track_dist_id[m].gperson_id << " dist" <<track_dist_id[m].eclu_distance<< endl;
                        }


                        int current_face_area = person_info[i].face_rect.width * person_info[i].face_rect.height;
                        int face_area_thresh  = current_face_area / 2;
                        for (int l = 0; l < track_dist_id.size() ; l++){
                            int g_face_area = g_person_track_info[track_dist_id[l].gperson_id].face_rect.width * g_person_track_info[track_dist_id[l].gperson_id].face_rect.height;
                            double area_diff = abs(g_face_area - current_face_area);
                   //         cout << " ########### area diff " << area_diff << endl;
                            if(area_diff < face_area_thresh){
                                //face_area_thresh = area_diff;
                                min_index = track_dist_id[l].gperson_id;
                                break;


                            }

                        }
                        if(min_index == -1){
                            min_index = track_dist_id[0].gperson_id;
                        }

                        g_person_track_info[min_index].cent.x = (person_info[i].face_rect.x + person_info[i].face_rect.width / 2);
                        g_person_track_info[min_index].cent.y = (person_info[i].face_rect.y + person_info[i].face_rect.height / 2);

                        g_person_track_info[min_index].face_rect.x = person_info[i].face_rect.x;
                        g_person_track_info[min_index].face_rect.y = person_info[i].face_rect.y;
                        g_person_track_info[min_index].face_rect.width = person_info[i].face_rect.width;
                        g_person_track_info[min_index].face_rect.height = person_info[i].face_rect.height;

                        g_person_track_info[min_index].person_rect = person_info[i].person_rect;

                        g_person_track_info[min_index].face_count = FACE_CNT;


                        g_person_track_info[min_index].frame_flag = true;					//cout << "#### test 6 bbbbb ###### size " <<g_person_track_info[min_index].classification_result_vector.size() << endl;

                        g_person_track_info[min_index].classification_result_vector.push_back(person_info[i].classification_result);					//cout << "#### test 6 ccccc###### " <<endl;



                        if(g_person_track_info[min_index].classification_result_vector.size() > PAST_CNT){

                            g_person_track_info[min_index].classification_result_vector.erase(g_person_track_info[min_index].classification_result_vector.begin() , g_person_track_info[min_index].classification_result_vector.begin() +g_person_track_info[min_index].classification_result_vector.size() - PAST_CNT - 1);
                        }


                        person_info[i].person_id   = g_person_track_info[min_index].person_id;

                        if(g_person_track_info[min_index].classification_result_vector.size() >= 1){
                            person_info[i].classification_result = g_person_track_info[min_index].classification_result_vector[0];
                            for(int k = 1; k < g_person_track_info[min_index].classification_result_vector.size(); k++){

                                if(person_info[i].classification_result != g_person_track_info[min_index].classification_result_vector[k]){
                                    person_info[i].classification_result = 0;
                                }
                            }
                        }
                        else{
                            person_info[i].classification_result = 0;

                        }


                        //track_result_info.push_back(temp_person);

                        push_person_track_result(person_info[i]);

                    }

                    if(min_index == -1){

                        person_info[i].person_id = person_id;
                        push_person_gtrack(person_info[i]);
                        push_person_track_result(person_info[i]);

                    }

                }
                else{

                    person_info[i].person_id = person_id;
                    push_person_gtrack(person_info[i]);
                    push_person_track_result(person_info[i]);

                }

            }
            else{

                person_info[i].classification_result = 0;
                push_person_track_result(person_info[i]);

            }

        }

    }



#if YOLO_DEBUG

    cout << "track_result_info_size " <<track_result_info.size() <<endl;
#endif

    return track_result_info;


}


///////////////// //////////// //////////////////////////////


#if 0

std::vector<roi_data> yolo_track (Mat frame, std::vector<roi_data> person_info){

    std::vector<person_track> temp_g_person_track_info;


    std::vector<roi_data> track_result_info;

    for (int j = 0; j < g_person_track_info.size(); j++){

        g_person_track_info[j].frame_flag = false;
        g_person_track_info[j].face_count --;

    }

#if DRAW_TRACK_RECT

#endif


    cout << "g_person_track_info.size_1 ::: " << g_person_track_info.size() << "size of person_info " << person_info.size()  << endl;
    if(g_person_track_info.size() > 0){
        if(person_info.size() > 0){

            for(int i = 0 ; i < person_info.size(); i++){


                //if((person_info[i].person_rect.x == 0 && person_info[i].person_rect.y == 0 && person_info[i].person_rect.width == 0 && person_info[i].person_rect.height == 0) || (person_info[i].face_rect.x == 0 && person_info[i].face_rect.y == 0 && person_info[i].face_rect.width == 0 && person_info[i].face_rect.height == 0) ){
                if((person_info[i].face_rect.x == 0 && person_info[i].face_rect.y == 0 && person_info[i].face_rect.width == 0 && person_info[i].face_rect.height == 0) ){

                    roi_data temp_person;
                    temp_person.face_rect = person_info[i].face_rect;
                    temp_person.person_rect = person_info[i].person_rect;
                    temp_person.person_id = person_id;
                    person_id++;

                    temp_person.classification_result = 0;
                    track_result_info.push_back(temp_person);
                }
                else{



                    std::vector<int> eclu_dist_vector;
                    std::vector<int> temp_j_vector;

                    std::vector<gperson_track_dist_id> track_dist_id;

                    cout << " ##### person_info[i].face_rect.width * 3 #### " << person_info[i].face_rect.width * 3 << endl;

                    int temp_dist = person_info[i].face_rect.width * 3;//eclu_dist_thresh;

                    int min_index = -1;int temp_j = -1;
                    for (int j = 0; j < g_person_track_info.size() ; j++){

                        //Calculate ecludian distance

                        if(g_person_track_info[j].frame_flag == false){

                            diff_x = g_person_track_info[j].cent.x - (person_info[i].face_rect.x + person_info[i].face_rect.width / 2);
                            diff_y = g_person_track_info[j].cent.y - (person_info[i].face_rect.y + person_info[i].face_rect.height / 2);

                            eclu_dist = pow(diff_x, 2) + pow(diff_y, 2);
                            eclu_dist = sqrt(eclu_dist);

                            if (temp_dist > eclu_dist){
                                gperson_track_dist_id temp_track_dist_id;
                                temp_track_dist_id.eclu_distance = eclu_dist;
                                temp_track_dist_id.gperson_id = j;


                                track_dist_id.push_back(temp_track_dist_id);
                                //temp_j = 0;

                                //eclu_dist_vector.push_back(eclu_dist);
                                //temp_j_vector.push_back(j);
                            }

                            //cout << "eclu dist " <<eclu_dist <<endl;
                        }
                        //////

                    }

                    //cout << "#### test 3 ###### " << "eclu_dist_vector size" << eclu_dist_vector.size() <<endl;
                    //sort track_dist_id with area
                    cout << "size of track_dist :" << track_dist_id.size() <<endl;

                    if(track_dist_id.size() > 0){
                        sort(track_dist_id.begin(), track_dist_id.end(), [](const gperson_track_dist_id a, const gperson_track_dist_id b) {return a.eclu_distance < b.eclu_distance; });
                        min_index = track_dist_id[0].gperson_id;

                        for (int m = 0;  m < track_dist_id.size(); m++){
                            cout << "track id" << track_dist_id[m].gperson_id << " dist" <<track_dist_id[m].eclu_distance<< endl;
                        }



                        int current_face_area = person_info[i].face_rect.width * person_info[i].face_rect.height;
                        int face_area_thresh  = current_face_area / 2;
                        for (int l = 0; l < track_dist_id.size() ; l++){
                            int g_face_area = g_person_track_info[track_dist_id[l].gperson_id].face_rect.width * g_person_track_info[track_dist_id[l].gperson_id].face_rect.height;
                            double area_diff = abs(g_face_area - current_face_area);
                            cout << " ########### area diff " << area_diff << endl;
                            if(area_diff < face_area_thresh){
                                face_area_thresh = area_diff;
                                min_index = track_dist_id[l].gperson_id;
                                break;


                            }

                        }

                    }
                    /*
                                                        for(int temp_i = 0;temp_i < track_dist_id.size(); temp_i++ ){

                                                                for(int temp_j = 0; temp_j < track_dist_id.size() ; temp_j++){
                                                                        track_dist_id[temp_i].eclu_distance
                                                                }

                                                        }
        */







                    /*
                                                int min_index = -1;
                                                int temp_j = -1;
                                                int current_face_area = person_info[i].face_rect.width * person_info[i].face_rect.height;
                                                int face_area_thresh  = current_face_area / 2;
                                                for(unsigned int k = 0; k < track_dist_id.size(); k++){

                                                                        temp_dist = track_dist_id[k].;
                                                                        min_index = k;
                                                                        temp_j = temp_j_vector[k];

                                                        }
*/



                    //cout << "#### test 4 ###### " << "temp_ dist " <<temp_dist <<endl;


                    if(temp_dist < eclu_dist_thresh && min_index != -1){
                        //cout << "min_index != -1" << endl;
                        //min_index = temp_j;

                        g_person_track_info[min_index].cent.x = (person_info[i].face_rect.x + person_info[i].face_rect.width / 2);
                        g_person_track_info[min_index].cent.y = (person_info[i].face_rect.y + person_info[i].face_rect.height / 2);

                        g_person_track_info[min_index].face_rect.x = person_info[i].face_rect.x;
                        g_person_track_info[min_index].face_rect.y = person_info[i].face_rect.y;
                        g_person_track_info[min_index].face_rect.width = person_info[i].face_rect.width;
                        g_person_track_info[min_index].face_rect.height = person_info[i].face_rect.height;

                        g_person_track_info[min_index].person_rect = person_info[i].person_rect;

                        g_person_track_info[min_index].face_count = FACE_CNT;


                        g_person_track_info[min_index].frame_flag = true;					//cout << "#### test 6 bbbbb ###### size " <<g_person_track_info[min_index].classification_result_vector.size() << endl;

                        g_person_track_info[min_index].classification_result_vector.push_back(person_info[i].classification_result);					//cout << "#### test 6 ccccc###### " <<endl;



                        if(g_person_track_info[min_index].classification_result_vector.size() > PAST_CNT){

                            g_person_track_info[min_index].classification_result_vector.erase(g_person_track_info[min_index].classification_result_vector.begin() , g_person_track_info[min_index].classification_result_vector.begin() +g_person_track_info[min_index].classification_result_vector.size() - PAST_CNT - 1);
                        }


                    }

                    else if(min_index == -1){
                        //cout << " adding ...................min_index == -1" << endl;
                        person_track person_track;

                        person_track.cent.x = person_info[i].face_rect.x + person_info[i].face_rect.width / 2;
                        person_track.cent.y = person_info[i].face_rect.y + person_info[i].face_rect.height / 2;

                        person_track.face_rect = person_info[i].face_rect;
                        person_track.person_rect = person_info[i].person_rect;
                        person_track.person_id = person_id;
                        //cout << "person_track.person_id : " << person_track.person_id <<endl;

                        person_id++;//cout << "####### 2222 person_id count ######## " << person_id << endl;

                        person_track.classification_result_vector.push_back(person_info[i].classification_result);
                        person_track.face_count = FACE_CNT;
                        person_track.frame_flag = true;

                        g_person_track_info.push_back(person_track);
                        min_index  = g_person_track_info.size() - 1;
                        //g_person_track_info[min_index].frame_flag = true;
                        //g_person_track_info[min_index].person_id = person_id;
                        //person_id ++;
                        //g_person_track_info[min_index].classification_result_vector.push_back(person_info[i].classification_result);

                    }


                    roi_data temp_person;
                    temp_person.face_rect = g_person_track_info[min_index].face_rect;
                    temp_person.person_rect = g_person_track_info[min_index].person_rect;
                    temp_person.person_id = g_person_track_info[min_index].person_id;
                    //cout << "temp_person.person_id" << temp_person.person_id << endl;

                    if(g_person_track_info[min_index].classification_result_vector.size() > 1 && g_person_track_info[min_index].classification_result_vector.size() >= PAST_CNT){
                        temp_person.classification_result = g_person_track_info[min_index].classification_result_vector[0];
                        for(int j = 1; j < g_person_track_info[min_index].classification_result_vector.size(); j++){

                            if(temp_person.classification_result != g_person_track_info[min_index].classification_result_vector[j]){
                                temp_person.classification_result = 0;
                            }
                            else{


                            }


                        }
                    }
                    else{
                        temp_person.classification_result = 0;


                    }


                    //temp_person.classification_result = g_person_track_info[min_index].classification_result_vector[classification_result_vector.size()];

                    //person_track;


                    //person_track.p

                    track_result_info.push_back(temp_person);

                }
            }



        }
    }
    else{

        if(person_info.size() > 0){
            //cout << "############# ELSE ###########" << endl;
            for(int i = 0 ; i < person_info.size(); i++){



                if((person_info[i].person_rect.x == 0 && person_info[i].person_rect.y == 0 && person_info[i].person_rect.width == 0 && person_info[i].person_rect.height == 0) || (person_info[i].face_rect.x == 0 && person_info[i].face_rect.y == 0 && person_info[i].face_rect.width == 0 && person_info[i].face_rect.height == 0) ){

                    roi_data temp_person;
                    temp_person.face_rect = person_info[i].face_rect;
                    temp_person.person_rect = person_info[i].person_rect;
                    temp_person.person_id = person_id;
                    person_id++;//cout << "####### 33333 person_id count ######## " << person_id << endl;
                    temp_person.classification_result = 0;
                    track_result_info.push_back(temp_person);
                }
                else{

                    person_track person_track;


                    person_track.face_count = FACE_CNT;

                    person_track.cent.x = person_info[i].face_rect.x + person_info[i].face_rect.width / 2;
                    person_track.cent.y = person_info[i].face_rect.y + person_info[i].face_rect.height / 2;

                    person_track.face_rect = person_info[i].face_rect;
                    person_track.person_rect = person_info[i].person_rect;

                    person_track.person_id = person_id;
                    //cout << "1111  person_track.person_id " << person_track.person_id << endl;
                    person_id++;//cout << "####### 444444 person_id count ######## " << person_id << endl;
                    person_track.frame_flag = true;

                    person_track.classification_result_vector.push_back(person_info[i].classification_result);


                    g_person_track_info.push_back(person_track);
                }
            }



            for (int j = 0; j < g_person_track_info.size() ; j++){
                roi_data temp_person;
                temp_person.face_rect   = g_person_track_info[j].face_rect;
                temp_person.person_rect = g_person_track_info[j].person_rect;
                temp_person.person_id   = g_person_track_info[j].person_id;
                //cout << "temp_person.person_id" << temp_person.person_id << endl;

                if(g_person_track_info[j].classification_result_vector.size() >= 1){
                    temp_person.classification_result = g_person_track_info[j].classification_result_vector[0];
                    for(int k = 1; k < g_person_track_info[j].classification_result_vector.size(); k++){
                        if(temp_person.classification_result != g_person_track_info[j].classification_result_vector[k]){
                            temp_person.classification_result = 0;
                        }
                        else{


                        }
                    }
                }
                else{
                    temp_person.classification_result = 0;


                }
                track_result_info.push_back(temp_person);

            }
        }
    }


    //##########

#if 1
    // TO REMOVE STRUCTURE IF FRAME
    if(g_person_track_info.size() > 0){
        cout << "size 2 of g_person_track_info " << g_person_track_info.size() << endl;
        for (int j = 0; j < g_person_track_info.size(); j++){

            //cout << " ################ person_id ################# "<< g_person_track_info[j].person_id << endl ;

            if(g_person_track_info[j].face_count > 0){
                temp_g_person_track_info.push_back(g_person_track_info[j]);
            }
        }

        g_person_track_info = temp_g_person_track_info;
    }

#endif

    return track_result_info;

}


#endif

std::vector<roi_data> yolo_darknet_classifier (std::vector<roi_data> person_info, cv::Mat MatOriginal){

#if 0		// uncomment for shivon 25/07/2018
    for(size_t i=0;i<person_info.size();i++)
    {

        if((person_info[i].face_rect.x == 0 && person_info[i].face_rect.y == 0 && person_info[i].face_rect.width == 0 && person_info[i].face_rect.height == 0) ){

            person_info[i].classification_result = 0;
        }
        else{





            Rect detect_face = person_info[i].face_rect;
            Rect detect_person = person_info[i].person_rect;

            int face_width = detect_face.width;
            int face_height = detect_face.height;

            int area = face_width * face_height;

            float ratio;
            if(face_height > 0)
            {
                ratio = (float)face_width/(float)face_height;
            }
            else
            {
                ratio = 0;
            }

            cout <<"!!!!!!!!!!!!!!!!!!!!aspect ratio" <<ratio<<endl;

            cout <<"!!!!!!!!!!!!!!!!!!!!face_width*********" <<face_width<<endl;
            cout <<"!!!!!!!!!!!!!!!!!!!!face_height ********" <<face_height<<endl;

            if(face_width > 20 && face_height > 25 && ratio > 0.3 && ratio < 1 && face_width < 90 && face_height < 200 )
            {

                if(area <= 9000 && area > 1000)
                {


                    //             can_count_faces = 1;

                    cv::Rect facerect = person_info[i].face_rect;

                    int height = int( facerect.height + facerect.width * 1.0 /5.0);

                    facerect.x = int(facerect.x  + facerect.width /2.0  - height /2.0);
                    facerect.y = int(facerect.y  + facerect.height /2.0 - height /2.0);
                    facerect.width = height;//int(facerect.height);// + facerect.width * 2.0 /3.0);
                    facerect.height = height;


                    if(facerect.x < 0)
                        facerect.x = 0;
                    if(facerect.y < 0)
                        facerect.y = 0;
                    if(facerect.y + facerect.height > MatOriginal.size().height)
                        facerect.height = facerect.height - ( (facerect.y + facerect.height) - MatOriginal.size().height);
                    if(facerect.x + facerect.width > MatOriginal.size().width)
                        facerect.width = facerect.width - (facerect.x + facerect.width - MatOriginal.size().width);

                    cv::Mat mat_crop = MatOriginal(facerect);
                    int face_index = darknet_class.darknet_classifier(mat_crop.clone());
                    person_info[i].classification_result = face_index;

                    cout<<"classifier values*********"<< person_info[i].classification_result<<endl;

                }

            }
        }



    }
#else
    for(int i = 0 ; i < person_info.size(); i++){


        if((person_info[i].face_rect.x == 0 && person_info[i].face_rect.y == 0 && person_info[i].face_rect.width == 0 && person_info[i].face_rect.height == 0) ){

            person_info[i].classification_result = 0;
        }
        else{
            /*
                                cv::Mat crop_face = MatOriginal(person_info[i].face_rect);
                                int cap_mask = darknet_class.darknet_classifier(crop_face);
                                person_info[i].classification_result = cap_mask;
                                cout << "CLASSIFICATION RESULT " << person_info[i].classification_result << endl;
*/

            cv::Rect facerect = person_info[i].face_rect;

            int height = int( facerect.height + facerect.width * 1.0 /5.0);

            facerect.x = int(facerect.x  + facerect.width /2.0  - height /2.0);
            facerect.y = int(facerect.y  + facerect.height /2.0 - height /2.0);
            facerect.width = height;//int(facerect.height);// + facerect.width * 2.0 /3.0);
            facerect.height = height;


            if(facerect.x < 0)
                facerect.x = 0;
            if(facerect.y < 0)
                facerect.y = 0;
            if(facerect.y + facerect.height > MatOriginal.size().height)
                facerect.height = facerect.height - ( (facerect.y + facerect.height) - MatOriginal.size().height);
            if(facerect.x + facerect.width > MatOriginal.size().width)
                facerect.width = facerect.width - (facerect.x + facerect.width - MatOriginal.size().width);

            cv::Mat mat_crop = MatOriginal(facerect);
            int face_index = darknet_class.darknet_classifier(mat_crop.clone());
            person_info[i].classification_result = face_index;

            cout << "CLASSIFICATION RESULT " << person_info[i].classification_result << endl;





        }


    }

#endif
    return person_info;
}


};


#endif
