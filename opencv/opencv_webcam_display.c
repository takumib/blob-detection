#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <cv.h>
#include <highgui.h>
#include <errno.h>
#include <sys/time.h>

#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>

#include "harris_cpu.hpp"
#include "harris_host.hpp"

#define DEBUG 1

#define QUIT 	  'q'
#define OP	 	  'o'
#define SH 	 	  's'
#define OP_SHARED 'z'
#define UN	 	  'u'

//Takes an image and blacks it out

// compiled with
//gcc example.c -o example `pkg-config --cflags --libs opencv`
// on my mac

// code source from http://www.cs.iit.edu/~agam/cs512/lect-notes/opencv-intro/


int main(int argc, char *argv[])
{
	printf("Begin\n");
	//Try and capture from a webcam
	CvCapture * capture = cvCaptureFromCAM(0);//CV_CAP_ANY);

	//If the capture failed, let the user know
	if(!capture){
		printf("Capture failed! %s\n", strerror(errno));
		return -1;
	}

	IplImage* img = cvQueryFrame(capture); 
	//cvCvtColor(img, img, CV_RGB2GRAY);
	int height,width,step,channels;
	unsigned char *data;
	int i,j,k;
	int key;

	// get the image data
	height    = img->height;
	width     = img->width;
	step      = img->widthStep;
	channels  = img->nChannels;
	data      = (unsigned char *)img->imageData;


	//Greyscaling code	
	IplImage *dest = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	cvCvtColor(img, dest, CV_RGB2GRAY);
	
	printf("Starting the webcam feed\n");
	printf("Processing a %dx%d image with %d channels\n",height,width,channels); 

	wakeup();
	struct timeval start;
	struct timeval end;
	float elapsedtime;

	int op = 0;

	while (key != QUIT){	

		img = cvQueryFrame(capture);
		dest = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		cvCvtColor(img, dest, CV_RGB2GRAY);
		
		data = (unsigned char *)dest->imageData;

		if(!img){
			break;
		}

		gettimeofday(&start, NULL);		
		
		if(op == 0)
			run_cpu(data, width, height, channels);
		else if(op == 1)
			run_gpu(data, width, height);
		else if(op == 2)
			run_gpu_op(data, width, height);
		else if(op == 3)
			run_gpu_shared_op(data, width, height);
		
		gettimeofday(&end, NULL);

		elapsedtime = (end.tv_sec - start.tv_sec) * 1000.0;
		elapsedtime += (end.tv_usec - start.tv_usec) / 1000.0;

		printf("FPS: %.2f\n", 1.0 / (elapsedtime / 100));

	
   		cvShowImage("mainWin", dest);
		key = cvWaitKey(1);

		switch(key) 
		{
			case OP:
				op = 1;
				break;
			case SH:
				op = 2;
				break;
			case OP_SHARED:
				op = 3;
				break;
			case UN:
				op = 0;
				break;
		}
	}



	//Do the computaiton here


	// release the image
	cvReleaseImage(&dest );
	return 0;

}


