// Backgroundsubstraction-Cuda.cpp : définit le point d'entrée pour l'application console.
//

#include "AxisCommunication.h"

#include "stdafx.h"

#include "cuda_runtime.h" 
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/types_c.h> 
#include <opencv2/imgproc/imgproc.hpp> 

using namespace cv;

extern "C" bool GPGPU_TstImg_CV_8U(cv::Mat* img, cv::Mat* GPGPUimg);


int defaultFgColor[3] = { 255, 255, 255 };
extern "C" bool GPGPU_BackGroundSubstractionHSV(cv::Mat* imgHSV, cv::Mat* GPGPUimg, int minHue, int maxHue,	
	int* backGroundColor, bool replaceForeground = false, int* ForegroundColor = defaultFgColor);

extern "C" bool GPGPU_Sobel(cv::Mat* imgHSV, cv::Mat* GPGPUimg, cv::Mat* Grayscale);

int main()

{
	Mat frame;
	Axis axis("10.128.3.4", "etudiant", "gty970");

	char* winName = "Trackbar";

	char* winFrame = "AXIS";
	namedWindow(winFrame, WINDOW_NORMAL);
	resizeWindow(winFrame, 1800, 900);
	char* winHSV = "HSV";
	namedWindow(winHSV, WINDOW_NORMAL);
	resizeWindow(winHSV, 1800, 900);

	namedWindow(winName);

	int minHue = 0;
	int maxHue = 179;
	createTrackbar("minHue", winName, &minHue, maxHue);
	createTrackbar("maxHue", winName, &maxHue, maxHue);

	//int minSat = 0;
	//int maxSat = 255;
	//createTrackbar("minSat", winName, &minSat, maxSat);
	//createTrackbar("maxSat", winName, &maxSat, maxSat);

	//int minVal = 0;
	//int maxVal = 255;
	//createTrackbar("minVal", winName, &minVal, maxVal);
	//createTrackbar("maxVal", winName, &maxVal, maxVal);

	while (true) {

		//frame = imread("../Pictures/lena.jpg");
		axis.GetImage(frame);

		if (frame.empty()) {
			break;
		}

		imshow(winFrame, frame);

		Mat imgHSV = frame.clone();
		Mat imgTresh = frame.clone();
		Mat imgSobel = frame.clone();

		Mat imgGrayscale;
		cvtColor(frame, imgGrayscale, CV_BGR2GRAY);

		GPGPU_TstImg_CV_8U(&frame, &imgHSV);
		imshow(winHSV, imgHSV);
	
		int bgColor[3] = { 0, 0, 0 };
		int fgColor[3] = { 255, 0, 0 };
		GPGPU_BackGroundSubstractionHSV(&imgHSV, &imgTresh, minHue, maxHue, bgColor, true);
		imshow("Treshold", imgTresh);

		/*GPGPU_Sobel(&imgTresh, &imgSobel, &imgGrayscale);
		imshow("Sobel", imgSobel);
		imshow("Grayscale", imgGrayscale);*/

		if (waitKey(30) >= 0) break; // Quit if key entered	
	}
	
	axis.ReleaseCam();

    return 0;
}

