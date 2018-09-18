#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "cuda_runtime.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "stdafx.h"

typedef unsigned char uchar;
typedef unsigned int uint;

#define BLOCK_SIZE 32
#define CV_64FC1 double
#define CV_32F float
#define CV_8U uchar

int iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__
double maxVal(double blue, double green, double red) {
	if ((blue >= green) && (blue >= red))
		return blue;
	else if ((green >= blue) && (green >= red))
		return green;
	else
		return red;
}

__device__
double minVal(double blue, double green, double red) {
	if ((blue <= green) && (blue <= red))
		return blue;
	else if ((green <= blue) && (green <= red))
		return green;
	else
		return red;
}

__global__ 
void Kernel_ThresholdHSV(uchar *img, uchar *imgout, int ImgWidth, int imgHeigh, int minHue, int maxHue, int* backGroundColor, bool replaceForeground, int* ForegroundColor)
{
	int ImgNumColonne = blockIdx.x  * blockDim.x + threadIdx.x;
	int ImgNumLigne = blockIdx.y  * blockDim.y + threadIdx.y;
	int Index = (ImgNumLigne * ImgWidth) + (ImgNumColonne * 3);

	if ((ImgNumColonne < ImgWidth / 3) && (ImgNumLigne < imgHeigh))
	{
		double blue = (double)img[Index] / 255;
		double green = (double)img[Index + 1] / 255;
		double red = (double)img[Index + 2] / 255;

		double cMax = maxVal(blue, green, red);
		double cMin = minVal(blue, green, red);

		double delta = cMax - cMin;

		//	HUE
		double hue = 0;
		if (blue == cMax) {
			hue = 60 * ((red - green) / delta + 4);
		}
		else if (green == cMax) {
			hue = 60 * ((blue - red) / delta + 2);
		}
		else if (red == cMax) {
			hue = 60 * ((green - blue) / delta);
			if (hue < 0)
				hue += 360;
		}

		//	SATURATION
		double saturation = 0;
		if (cMax != 0) {
			saturation = delta / cMax;
		}

		//	VALUE
		double value = cMax;

		hue = hue / 2;	// 0-179
		saturation = saturation * 255; // 0-255
		value = value * 255; // 0-255

		if (hue > minHue && hue < maxHue && value >= 130) {	//	Background-Green is black by default
			imgout[Index] = backGroundColor[0];
			imgout[Index + 1] = backGroundColor[1];
			imgout[Index + 2] = backGroundColor[2];
		}
		else {
			if (replaceForeground) {
				imgout[Index] = ForegroundColor[0];
				imgout[Index + 1] = ForegroundColor[1];
				imgout[Index + 2] = ForegroundColor[2];
			}
			else {
				imgout[Index] = img[Index];
				imgout[Index + 1] = img[Index + 1];
				imgout[Index + 2] = img[Index + 2];
			}		
		}
	}

	return;
}

int defaultForegroundColor[3] = { 255, 255, 255 };
extern "C" bool GPGPU_BackGroundSubstractionHSV(cv::Mat* imgHSV, cv::Mat* GPGPUimg, int minHue, int maxHue,
	int* backGroundColor, bool replaceForeground = false, int* ForegroundColor = defaultForegroundColor)
{
	//	1. Initialize data
	cudaError_t cudaStatus;
	uchar *gDevImage;
	uchar *gDevImageOut;
	int* gBgColor;
	int* gFgColor;

	uint imageSize = imgHSV->rows * imgHSV->step1();
	uint ColorSize = sizeof(int) * 3;

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(iDivUp(imgHSV->cols, BLOCK_SIZE), iDivUp(imgHSV->rows, BLOCK_SIZE));

	//	2. Allocation data
	cudaStatus = cudaMalloc(&gDevImage, imageSize);
	cudaStatus = cudaMalloc(&gDevImageOut, imageSize);
	cudaStatus = cudaMalloc(&gBgColor, ColorSize);
	cudaStatus = cudaMalloc(&gFgColor, ColorSize);

	//	3. Copy data on GPU
	cudaStatus = cudaMemcpy(gDevImage, imgHSV->data, imageSize, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(gBgColor, backGroundColor, ColorSize, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(gFgColor, ForegroundColor, ColorSize, cudaMemcpyHostToDevice);

	//	4. Launch kernel
	Kernel_ThresholdHSV << <dimGrid, dimBlock >> >(gDevImage, gDevImageOut, imgHSV->step1(), imgHSV->rows, minHue, maxHue, gBgColor, replaceForeground, gFgColor);	

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	//Wait for the kernel to end
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize failed!");
		goto Error;
	}

	//	5. Copy data on CPU
	cudaStatus = cudaMemcpy(GPGPUimg->data, gDevImageOut, imageSize, cudaMemcpyDeviceToHost);

	//	6. Free GPU memory
Error:
	cudaFree(gDevImage);
	cudaFree(gDevImageOut);

	return cudaStatus;
}

__device__
int absGrad(int grad) {
	if (grad < 0) {
		return -1 * grad;
	}
	else {
		return grad;
	}
}

__global__ 
void Kernel_Sobel(uchar* img, uchar* imgout, int ImgWidth, int imgHeigh) // , int* maskX, int* maskY
{	
	int ImgNumColonne = blockIdx.x  * blockDim.x + threadIdx.x;
	int ImgNumLigne = blockIdx.y  * blockDim.y + threadIdx.y;

	int Index = (ImgNumLigne * ImgWidth) + (ImgNumColonne * 3);
	int IndexGray = (ImgNumLigne * (ImgWidth / 3)) + (ImgNumColonne);

	if ((ImgNumColonne < (ImgWidth / 3) - 2) && (ImgNumLigne < imgHeigh - 2)) {

		int i = Index;
		int gradX = img[i] * -3 + img[i + 3] * 0 + img[i + 6] * 3;
		i = ((ImgNumLigne + 1) * ImgWidth) + (ImgNumColonne * 3);
		gradX += img[i] * -10 + img[i + 3] * 0 + img[i + 6] * 10;
		i = ((ImgNumLigne + 2) * ImgWidth) + (ImgNumColonne * 3);
		gradX += img[i] * -3 + img[i + 3] * 0 + img[i + 6] * 3;

		i = Index;
		int gradY = img[i] * -3 + img[i + 3] * -10 + img[i + 6] * -3;
		i = ((ImgNumLigne + 1) * ImgWidth) + (ImgNumColonne * 3);
		gradY += img[i] * 0 + img[i + 3] * 0 + img[i + 6] * 0;
		i = ((ImgNumLigne + 2) * ImgWidth) + (ImgNumColonne * 3);
		gradY += img[i] * 3 + img[i + 3] * 10 + img[i + 6] * 3;


		int grad = absGrad(gradX) + absGrad(gradY);
		int norm = grad * 0.0625;

		imgout[IndexGray] = norm;

	}

	return;
}

extern "C" bool GPGPU_Sobel(cv::Mat* imgTresh, cv::Mat* Grayscale)
{
	//	1. Initialize data
	cudaError_t cudaStatus;
	uchar* gDevImage;
	uchar* gDevImageOut;

	uint imageSize = imgTresh->rows * imgTresh->step1(); //	3x greater than gradientSize
	uint gradientSize = imgTresh->rows * imgTresh->cols * sizeof(uchar); 

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(iDivUp(imgTresh->cols, BLOCK_SIZE), iDivUp(imgTresh->rows, BLOCK_SIZE));

	//	2. Allocation data
	cudaStatus = cudaMalloc(&gDevImage, imageSize);
	cudaStatus = cudaMalloc(&gDevImageOut, gradientSize);

	//	3. Copy data on GPU
	cudaStatus = cudaMemcpy(gDevImage, imgTresh->data, imageSize, cudaMemcpyHostToDevice);

	//	4. Launch kernel
	Kernel_Sobel << <dimGrid, dimBlock >> >(gDevImage, gDevImageOut, imgTresh->step1(), imgTresh->rows);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	//Wait for the kernel to end
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize failed!");
		goto Error;
	}

	//	5. Copy data on CPU
	cudaStatus = cudaMemcpy(Grayscale->data, gDevImageOut, gradientSize, cudaMemcpyDeviceToHost);

	//	6. Free GPU memory
Error:
	cudaFree(gDevImage);
	cudaFree(gDevImageOut);

	return cudaStatus;
}
