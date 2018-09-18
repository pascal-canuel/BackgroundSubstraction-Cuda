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
#include <chrono> // for high_resolution_clock

using namespace cv;

extern "C" bool GPGPU_TstImg_CV_8U(cv::Mat* img, cv::Mat* GPGPUimg);

int defaultFgColor[3] = { 255, 255, 255 };
extern "C" bool GPGPU_BackGroundSubstractionHSV(cv::Mat* imgHSV, cv::Mat* GPGPUimg, int minHue, int maxHue,	
	int* backGroundColor, bool replaceForeground = false, int* ForegroundColor = defaultFgColor);

extern "C" bool GPGPU_Sobel(cv::Mat* imgHSV, cv::Mat* Grayscale);

int main()

{
	Mat frame;
	Axis axis("10.128.3.4", "etudiant", "gty970");

	char* winName = "Trackbar";
	namedWindow(winName);

	int lowGreen = 38;
	int highGreen = 75;
	int maxHue = 179;

	createTrackbar("minHue", winName, &lowGreen, maxHue);
	createTrackbar("maxHue", winName, &highGreen, maxHue);

	while (true) {

		axis.GetImage(frame);
		//frame = imread("../Pictures/3.jpg");

		if (frame.empty()) {
			break;
		}

		imshow("AXIS", frame);

		Mat imgHSV = frame.clone();
		Mat imgTresh = frame.clone();

		Mat imgGrayscale;
		cvtColor(frame, imgGrayscale, CV_BGR2GRAY);
		
		int bgColor[3] = { 0, 0, 0 };
		//int fgColor[3] = { 0, 0, 255 };
		GPGPU_BackGroundSubstractionHSV(&imgHSV, &imgTresh, lowGreen, highGreen, bgColor, false);
		imshow("Treshold", imgTresh);

		GPGPU_Sobel(&imgTresh, &imgGrayscale);
		imshow("Sobel", imgGrayscale);

		//waitKey(0); // Wait for key entered
		if (waitKey(30) >= 0) break; // Quit if key entered	
	}

    return 0;
}

