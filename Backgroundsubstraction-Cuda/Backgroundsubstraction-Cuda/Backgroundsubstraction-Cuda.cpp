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

int main()

{
	Mat Image;

	Axis axis("10.128.3.4", "etudiant", "gty970");

	axis.GetImage(Image);

	imshow("Axis PTZ", Image);

	waitKey(5);

	axis.ReleaseCam();

    return 0;
}

