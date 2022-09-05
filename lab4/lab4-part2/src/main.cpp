#include <iostream>
#include <string>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "img_proc.h"
#include "timer.h"

#define OPENCV 0
#define CPU 1
#define GPU 2

#define DO_GRAY 0
#define DO_INV 1
#define DO_BLUR 2

#define CUR_OPERATION DO_BLUR

#define BLUR_SIZE 10

#define UNIFIED_MEM 

using namespace std;
using namespace cv;

int usage()
{
	cout << "Usage: ./lab4 <mode> <WIDTH> <HEIGHT>" <<endl;
	cout << "mode: 0 OpenCV" << endl;
	cout << "      1 CPU" << endl;
	cout << "      2 GPU" << endl;
	return 0;
}

int use_mode(int mode)
{
	string descr;
	switch(mode)
	{
		case OPENCV:
			descr = "OpenCV Functions";
			break;
		case CPU:
			descr = "CPU Implementations";
			break;
		case GPU:
			descr = "GPU Implementations";
			break;
		default:
			descr = "None";
			return usage();
	}	
	
	cout << "Using " << descr.c_str() <<endl;
	return 1;
}

int main(int argc, const char *argv[]) 
{

	int mode = 0;

	if(argc >= 2)
	{
		mode = atoi(argv[1]);	
	}
	
	if(use_mode(mode) == 0)
		return 0;

	VideoCapture cap("input.raw");

	int WIDTH  = 768;
	int HEIGHT = 768;
	int CHANNELS = 3;

	// 1 argument on command line: WIDTH = HEIGHT = arg
	if(argc >= 3)
	{
		WIDTH = atoi(argv[2]);
		HEIGHT = WIDTH;
	}
	// 2 arguments on command line: WIDTH = arg1, HEIGHT = arg2
	if(argc >= 4)
	{
		HEIGHT = atoi(argv[3]);
	}

	// Profiling framerate
	LinuxTimer timer;
	LinuxTimer fps_counter;
	double time_elapsed = 0;

    uchar* dev_gray, *dev_rgb, *dev_output;
    int size = WIDTH*HEIGHT*CHANNELS*sizeof(uchar);
    int size_gray = WIDTH*HEIGHT*sizeof(uchar);
    

#ifndef UNIFIED_MEM
    //TODO: Allocate memory on the GPU device.
    //TODO: Declare the host image result matrices
    
    Mat rgb = Mat(HEIGHT, WIDTH, CV_8UC3);
	Mat gray = Mat(HEIGHT, WIDTH, CV_8U);
    Mat output = Mat(HEIGHT, WIDTH, CV_8U);
    
    cudaMalloc((void**)&dev_gray, size_gray);
    cudaMalloc((void**)&dev_output, size_gray);
    cudaMalloc((void**)&dev_rgb, size);
    
#else
    //TODO: Allocate unified memory for the necessary matrices
    //TODO: Declare the image matrices which point to the unified memory
    cudaMallocManaged(&dev_gray, size_gray);
    cudaMallocManaged(&dev_output, size_gray);
    cudaMallocManaged(&dev_rgb, size);
    
    Mat rgb = Mat(Size(WIDTH, HEIGHT), CV_8UC3, dev_rgb);
	Mat gray = Mat(Size(WIDTH, HEIGHT), CV_8U, dev_gray);
    Mat output = Mat(Size(WIDTH, HEIGHT), CV_8U, dev_output);
#endif
	
    

	//Matrix for OpenCV inversion
	Mat ones = Mat::ones(HEIGHT, WIDTH, CV_8U)*255;

	Mat frame;	
	char key=0;
	int count = 0;
    
    
	while (key != 'q')
	{
		cap >> frame;
		if(frame.empty())
		{
			waitKey();
			break;
		}

		resize(frame, rgb, Size(WIDTH, HEIGHT));

		imshow("Original", rgb);
        
        if(CUR_OPERATION != DO_GRAY)
        {
            // the other two operations use gray as a baseline, so generate that now
#ifdef OPENCV4
            cvtColor(rgb, gray, COLOR_BGR2GRAY);
#else
            cvtColor(rgb, gray, CV_BGR2GRAY);
#endif
        }

		timer.start();
		switch(mode)
		{
			case OPENCV:
                
                if(CUR_OPERATION == DO_GRAY)
                {
#ifdef OPENCV4
                    cvtColor(rgb, gray, COLOR_BGR2GRAY);
#else
                    cvtColor(rgb, gray, CV_BGR2GRAY);
#endif
                }
                else if (CUR_OPERATION == DO_INV)
                {
                    output = ones - gray;
                }
                else if (CUR_OPERATION == DO_BLUR)
                {
                    blur(gray,output,Size(BLUR_SIZE,BLUR_SIZE)); 
                }

				break;
			case CPU:
                // TODO: 1) Call the CPU functions
                
                
                if(CUR_OPERATION == DO_GRAY)
                {
                    img_rgb2gray_cpu(gray.ptr<uchar>(), rgb.ptr<uchar>(), WIDTH, HEIGHT, CHANNELS);
                }
                else if (CUR_OPERATION == DO_INV)
                {
                    img_invert_cpu(output.ptr<uchar>(), gray.ptr<uchar>(), WIDTH, HEIGHT);
                }
                else if (CUR_OPERATION == DO_BLUR)
                {
                    img_blur_cpu(output.ptr<uchar>(), gray.ptr<uchar>(), WIDTH, HEIGHT, BLUR_SIZE);
                }
				break;

			case GPU:   
#ifndef UNIFIED_MEM
                /* TODO: 1) Copy data from host to device
                 *       2) Call GPU host function with device data
                 *       3) Copy data from device to host
                */
                
                if(CUR_OPERATION == DO_GRAY)
                {
                    cudaMemcpy(dev_rgb, rgb.ptr<uchar>(), size, cudaMemcpyHostToDevice);
                    img_rgb2gray(dev_gray, dev_rgb, HEIGHT, WIDTH, CHANNELS);
                    int result = cudaDeviceSynchronize();
                    if(result != 0)
                    {
                        cout << "Grayscale failed with result: " << result << endl;
                    }
                    cudaMemcpy(gray.ptr<uchar>(), dev_gray, size_gray, cudaMemcpyDeviceToHost);
                }
                else if (CUR_OPERATION == DO_INV)
                {
                    cudaMemcpy(dev_gray, gray.ptr<uchar>(), size_gray, cudaMemcpyHostToDevice);
                    img_invert(dev_output, dev_gray, HEIGHT, WIDTH);
                    int result = cudaDeviceSynchronize();
                    if(result != 0)
                    {
                        cout << "Invert failed with result: " << result << endl;
                    }
                    cudaMemcpy(output.ptr<uchar>(), dev_output, size_gray, cudaMemcpyDeviceToHost);
                }
                else if (CUR_OPERATION == DO_BLUR)
                {
                    cudaMemcpy(dev_gray, gray.ptr<uchar>(), size_gray, cudaMemcpyHostToDevice);
                    img_blur(dev_output, dev_gray, HEIGHT, WIDTH, BLUR_SIZE);
                    int result = cudaDeviceSynchronize();
                    if(result != 0)
                    {
                        cout << "Invert failed with result: " << result << endl;
                    }
                    cudaMemcpy(output.ptr<uchar>(), dev_output, size_gray, cudaMemcpyDeviceToHost);
                }
                

#else
                if(CUR_OPERATION == DO_GRAY)
                {
                    img_rgb2gray(dev_gray, dev_rgb, HEIGHT, WIDTH, CHANNELS);
                    int result = cudaDeviceSynchronize();
                    cudaError_t result2 = cudaGetLastError();
                    if(result2 != 0)
                    {
                        cout << "Grayscale failed with result: " << cudaGetErrorString(result2) << endl;
                    }
                }
                else if (CUR_OPERATION == DO_INV)
                {
                    img_invert(dev_output, dev_gray, HEIGHT, WIDTH);
                    int result = cudaDeviceSynchronize();
                    if(result != 0)
                    {
                        cout << "Invert failed with result: " << result << endl;
                    }
                }
                else if (CUR_OPERATION == DO_BLUR)
                {
                    img_blur(dev_output, dev_gray, HEIGHT, WIDTH, BLUR_SIZE);
                    int result = cudaDeviceSynchronize();
                    if(result != 0)
                    {
                        cout << "Invert failed with result: " << result << endl;
                    }
                }
#endif
				break;
		}
		timer.stop();
        
        
		size_t time_rgb2gray = timer.getElapsed();
		
		count++;
		time_elapsed += (timer.getElapsed())/10000000000.0;

		if (count % 10 == 0)
		{
			cout << "Execution Time (s) = " << time_elapsed << endl;
			time_elapsed = 0;
		}

        if(CUR_OPERATION == DO_GRAY)
        {
            imshow("Gray", gray);
        }
        else
        {
            imshow("Output", output);
        }

		key = waitKey(1);
	}
    
    cudaFree(dev_gray);
    cudaFree(dev_rgb);
}
