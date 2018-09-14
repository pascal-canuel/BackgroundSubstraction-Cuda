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

//	TODO create type instead of scalar
extern "C" bool GPGPU_BackGroundSubstractionHSV(cv::Mat* imgHSV, cv::Mat* GPGPUimg, int minHue, int maxHue,
	cv::Scalar backGroundColor, bool replaceForeground = false, cv::Scalar ForegroundColor = cv::Scalar(0, 0, 0));

extern "C" bool GPGPU_Sobel(cv::Mat* imgHSV, cv::Mat* GPGPUimg, cv::Mat* Grayscale);

int main()

{
	Mat frame;
	Axis axis("10.128.3.4", "etudiant", "gty970");
	axis.GetImage(frame);

	char* winName = "Trackbar";
	char* winFrame = "AXIS";
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

		//frame = imread("../Pictures/lena.png");

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
		imshow("HSV", imgHSV);

		GPGPU_BackGroundSubstractionHSV(&imgHSV, &imgTresh, minHue, maxHue, Scalar(255, 255, 255));
		imshow("Treshold", imgTresh);

		GPGPU_Sobel(&imgTresh, &imgSobel, &imgGrayscale);
		imshow("Sobel", imgSobel);
		//imshow("Grayscale", imgGrayscale);

		if (waitKey(30) >= 0) break; // Quit if key entered	
	}
	
	axis.ReleaseCam();

    return 0;
}

