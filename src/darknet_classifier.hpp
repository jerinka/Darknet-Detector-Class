#ifndef DARKNET_CLASSIFIER_H
#define DARKNET_CLASSIFIER_H

#if 1
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cuda.h"
#endif

#include<iostream>
#include<opencv2/opencv.hpp>

extern "C"{
#include "parser.h"
#include "option_list.h"
#include "utils.h"
#include "network.h"
}


class Darknet_Classifier{

public :
	network net;
     list *options;

	char *name_list;
    
    int classes;
	int i;
	char **names;
	int *indexes;			// 0: capnomaskno 1:capnomaskyes 2:capyesmaskno 3:capyesmaskyes
	char buff[256];
	char *input;
	int size;
	int namesSize; 
	
	float *predictions;
    int top;
	char *filename=NULL;
	//IplImage* im;
    

	Darknet_Classifier(char *datacfg, char *cfgfile, char *weightfile){

		this->top = 0;
		net = parse_network_cfg_custom(cfgfile, 1);

    		if(weightfile){
        		load_weights(&net, weightfile);
    		}
    		set_batch_network(&net, 1);
			srand(2222222);

		options = read_data_cfg(datacfg);
		name_list = option_find_str(options, "names", 0);
		if(!name_list){ name_list = option_find_str(options, "labels", "../../data/capmask/labels.list");
			printf("#####%%%%%%%%%%%%%%%%%% \n");
		}
		classes = option_find_int(options, "classes", 2);
		if (top == 0) top = option_find_int(options, "top", 1);
		if (top > classes) top = classes;
		i = 0;
		//this->top = top;
		 
 
    		names = get_labels(name_list, &namesSize);
    		indexes = (int *)calloc(top, sizeof(int));
    
    		input = buff;
    		size = net.w;

	}
    Darknet_Classifier(){
		printf("DEFAULT CONSTRUCTOR \n");
    }				 
	
	~Darknet_Classifier()
	{
		free(indexes);
		free_network(net);

		if(options != NULL)
			free_list_contents_kvp(options);
//		if(options != NULL)
//			free_list_contents(options);
		if(options != NULL)
			free_list(options);
		for(int i = 0; i < namesSize; ++i)
			if(names[i] != NULL)
			free(names[i]);
		if(names != NULL)
		free(names);
//		if(name_list != NULL)
//		free(name_list);

		std::cout << "Darknet_Classifier Destructor" << std::endl;
	}


	int darknet_classifier(cv::Mat img1){

		cv::Mat img = img1.clone();
        cv::cvtColor(img, img, CV_BGR2RGB);
		cv::resize(img, img, cv::Size(50,50));
		IplImage* image2 = new IplImage(img);

		image im = ipl_to_image(image2);
		delete image2;
		
		
		//image im = load_image_color("capnomaskno.jpg", 0, 0);
		image r = letterbox_image(im, net.w, net.h);
		//image r = resize_min(im, size);
		//resize_network(&net, r.w, r.h);
		//printf("%d %d\n", r.w, r.h);

		float *X = r.data;
		predictions = network_predict(net, X);
		if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
		top_k(predictions, net.outputs, top, indexes);

		for(i = 0; i < top; ++i){
		    int index = indexes[i];
		    //if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
		    //else printf("%s: %f\n",names[index], predictions[index] * 100);
		}
		
		if(r.data != NULL) //if(r.data != im.data)
			free_image(r);
		if (im.data != NULL)
		    free_image(im);
		int indx = indexes[0];
		return indx;
		

	}



	image ipl_to_image(IplImage* src)
	{
	    unsigned char *data = (unsigned char *)src->imageData;
	    int h = src->height;
	    int w = src->width;
	    int c = src->nChannels;
	    int step = src->widthStep;
	    image out = make_image(w, h, c);
	    int i, j, k, count=0;;

	    for(k= 0; k < c; ++k){
		for(i = 0; i < h; ++i){
		    for(j = 0; j < w; ++j){
		        out.data[count++] = data[i*step + j*c + k]/255.;
		    }
		}
	    }
	    return out;
	}



	void darknet_classifier_train(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show){


		float avg_loss = -1;
		char *base = basecfg(cfgfile);
		printf("%s\n", base);
		printf("%d\n", ngpus);
		network *nets = (network *)calloc(ngpus, sizeof(network));

		srand(time(0));
		int seed = rand();
		for (int i = 0; i < ngpus; ++i) {
			srand(seed);
#ifdef GPU
			cuda_set_device(gpus[i]);
#endif
			nets[i] = parse_network_cfg(cfgfile);
			if (weightfile) {
				load_weights(&nets[i], weightfile);
			}
			if (clear) *nets[i].seen = 0;
			nets[i].learning_rate *= ngpus;
		}
		srand(time(0));
		network net = nets[0];

		int imgs = net.batch * net.subdivisions * ngpus;

		printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
		list *options = read_data_cfg(datacfg);

		char *backup_directory = option_find_str(options, "backup", "/backup/");
		char *label_list = option_find_str(options, "labels", "data/labels.list");
		char *train_list = option_find_str(options, "train", "data/train.list");
		int classes = option_find_int(options, "classes", 2);

		char **labels = get_labels(label_list, NULL);
		list *plist = get_paths(train_list);
		char **paths = (char **)list_to_array(plist);
		printf("%d\n", plist->size);
		int N = plist->size;
		clock_t time;

		load_args args = { 0 };
		args.w = net.w;
		args.h = net.h;
		args.threads = 32;
		args.hierarchy = net.hierarchy;

		args.min = net.min_crop;
		args.max = net.max_crop;
		args.flip = net.flip;
		args.angle = net.angle;
		args.aspect = net.aspect;
		args.exposure = net.exposure;
		args.saturation = net.saturation;
		args.hue = net.hue;
		args.size = net.w;

		args.paths = paths;
		args.classes = classes;
		args.n = imgs;
		args.m = N;
		args.labels = labels;
		args.type = CLASSIFICATION_DATA;

#ifdef OPENCV
		args.threads = 3;
		IplImage* img = NULL;
		float max_img_loss = 5;
		int number_of_lines = 100;
		int img_size = 1000;
		if (!dont_show)
			img = draw_train_chart(max_img_loss, net.max_batches, number_of_lines, img_size);
#endif  //OPENCV

		data train;
		data buffer;
		pthread_t load_thread;
		args.d = &buffer;
		load_thread = load_data(args);

		int iter_save = get_current_batch(net);
		while (get_current_batch(net) < net.max_batches || net.max_batches == 0) {
			time = clock();

			pthread_join(load_thread, 0);
			train = buffer;
			load_thread = load_data(args);

			printf("Loaded: %lf seconds\n", sec(clock() - time));
			time = clock();

			float loss = 0;
#ifdef GPU
			if (ngpus == 1) {
				loss = train_network(net, train);
			}
			else {
				loss = train_networks(nets, ngpus, train, 4);
			}
#else
			loss = train_network(net, train);
#endif
			if (avg_loss == -1) avg_loss = loss;
			avg_loss = avg_loss*.9 + loss*.1;

			i = get_current_batch(net);

			printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen) / N, loss, avg_loss, get_current_rate(net), sec(clock() - time), *net.seen);
#ifdef OPENCV
			if (!dont_show)
				draw_train_loss(img, img_size, avg_loss, max_img_loss, i, net.max_batches);
#endif  // OPENCV

			if (i >= (iter_save + 100)) {
				iter_save = i;
#ifdef GPU
				if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif            
				char buff[256];
				sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
				save_weights(net, buff);
			}
			free_data(train);
		}
#ifdef GPU
		if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif    
		char buff[256];
		sprintf(buff, "%s/%s_final.weights", backup_directory, base);
		save_weights(net, buff);

#ifdef OPENCV
		cvReleaseImage(&img);
		cvDestroyAllWindows();
#endif

		free_network(net);
		free_ptrs((void**)labels, classes);
		free_ptrs((void**)paths, plist->size);
		free_list(plist);
		free(base);

	}



	IplImage* draw_train_chart(float max_img_loss, int max_batches, int number_of_lines, int img_size)
	{
		int img_offset = 50;
		int draw_size = img_size - img_offset;
		IplImage* img = cvCreateImage(cvSize(img_size, img_size), 8, 3);
		cvSet(img, CV_RGB(255, 255, 255), 0);
		CvPoint pt1, pt2, pt_text;
		CvFont font;
		cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.7, 0.7, 0, 1, CV_AA);
		char char_buff[100];
		int i;
		// vertical lines
		pt1.x = img_offset; pt2.x = img_size, pt_text.x = 10;
		for (i = 1; i <= number_of_lines; ++i) {
			pt1.y = pt2.y = (float)i * draw_size / number_of_lines;
			cvLine(img, pt1, pt2, CV_RGB(224, 224, 224), 1, 8, 0);
			if (i % 10 == 0) {
				sprintf(char_buff, "%2.1f", max_img_loss*(number_of_lines - i) / number_of_lines);
				pt_text.y = pt1.y + 5;
				cvPutText(img, char_buff, pt_text, &font, CV_RGB(0, 0, 0));
				cvLine(img, pt1, pt2, CV_RGB(128, 128, 128), 1, 8, 0);
			}
		}
		// horizontal lines
		pt1.y = draw_size; pt2.y = 0, pt_text.y = draw_size + 15;
		for (i = 0; i <= number_of_lines; ++i) {
			pt1.x = pt2.x = img_offset + (float)i * draw_size / number_of_lines;
			cvLine(img, pt1, pt2, CV_RGB(224, 224, 224), 1, 8, 0);
			if (i % 10 == 0) {
				sprintf(char_buff, "%d", max_batches * i / number_of_lines);
				pt_text.x = pt1.x - 20;
				cvPutText(img, char_buff, pt_text, &font, CV_RGB(0, 0, 0));
				cvLine(img, pt1, pt2, CV_RGB(128, 128, 128), 1, 8, 0);
			}
		}
		cvPutText(img, "Iteration number", cvPoint(draw_size / 2, img_size - 10), &font, CV_RGB(0, 0, 0));
		cvPutText(img, "Press 's' to save: chart.jpg", cvPoint(5, img_size - 10), &font, CV_RGB(0, 0, 0));
		printf(" If error occurs - run training with flag: -dont_show \n");
		cvNamedWindow("average loss", CV_WINDOW_NORMAL);
		cvMoveWindow("average loss", 0, 0);
		cvResizeWindow("average loss", img_size, img_size);
		cvShowImage("average loss", img);
		cvWaitKey(20);
		return img;
	}

	void draw_train_loss(IplImage* img, int img_size, float avg_loss, float max_img_loss, int current_batch, int max_batches)
	{
		int img_offset = 50;
		int draw_size = img_size - img_offset;
		CvFont font;
		cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.7, 0.7, 0, 1, CV_AA);
		char char_buff[100];
		CvPoint pt1, pt2;
		pt1.x = img_offset + draw_size * (float)current_batch / max_batches;
		pt1.y = draw_size * (1 - avg_loss / max_img_loss);
		if (pt1.y < 0) pt1.y = 1;
		cvCircle(img, pt1, 1, CV_RGB(0, 0, 255), CV_FILLED, 8, 0);

		sprintf(char_buff, "current avg loss = %2.4f", avg_loss);
		pt1.x = img_size / 2, pt1.y = 30;
		pt2.x = pt1.x + 250, pt2.y = pt1.y + 20;
		cvRectangle(img, pt1, pt2, CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
		pt1.y += 15;
		cvPutText(img, char_buff, pt1, &font, CV_RGB(0, 0, 0));
		cvShowImage("average loss", img);
		int k = cvWaitKey(20);
		if (k == 's' || current_batch == (max_batches - 1)) cvSaveImage("chart.jpg", img, 0);
	}




};

#endif
