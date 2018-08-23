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
	float *X;
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
		if(!name_list){ name_list = option_find_str(options, "labels", "data/capmask/labels.list");
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


	int darknet_classifier(cv:: Mat img){


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

		X = r.data;
		predictions = network_predict(net, X);
		if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
		top_k(predictions, net.outputs, top, indexes);

		for(i = 0; i < top; ++i){
		    int index = indexes[i];
		    if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
		    else printf("%s: %f\n",names[index], predictions[index] * 100);
		}
		
		if(r.data != im.data)
			free_image(r);
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



};

#endif
