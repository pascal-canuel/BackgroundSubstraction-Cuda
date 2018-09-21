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

int defaultFgColor[3] = { 255, 255, 255 };
extern "C" bool GPGPU_BackGroundSubstractionHSV(cv::Mat* imgHSV, cv::Mat* GPGPUimg, int minHue, int maxHue,	
	int* backGroundColor, bool replaceForeground = false, int* ForegroundColor = defaultFgColor);

extern "C" bool GPGPU_Sobel(cv::Mat* imgHSV, cv::Mat* Grayscale);

int main()

{
	Mat frame;
	Axis axis("10.128.3.4", "etudiant", "gty970");

	char* winName = "AXIS";
	namedWindow(winName);

	int lowHue = 75;
	int highHue = 150;
	int max = 359;

	createTrackbar("minHue", winName, &lowHue, max);
	createTrackbar("maxHue", winName, &highHue, max);

	while (true) {
		//	Get actual frame from axis cam
		axis.GetImage(frame);
		//frame = imread("../Pictures/1.jpg");

		if (frame.empty()) {
			break;
		}

		imshow(winName, frame);

		//	Load another mat for sobel display
		Mat imgGrayscale;
		cvtColor(frame, imgGrayscale, CV_BGR2GRAY);
		
		int bgColor[3] = { 0, 0, 0 };
		int fgColor[3] = { 0, 0, 255 };
		GPGPU_BackGroundSubstractionHSV(&frame, &frame, lowHue, highHue, bgColor, true);
		imshow("Treshold", frame);

		GPGPU_Sobel(&frame, &imgGrayscale);
		imshow("Sobel", imgGrayscale);

		if (waitKey(5) >= 0) break; // Quit if key entered	
	}

    return 0;
}

