#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "cuda_runtime.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "stdafx.h"
//#include "nppdefs.h"
//#include <npp.h>

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

// Transfert img to imgout to see how opencv image can be acces in GPGPU
__global__ void Kernel_Tst_Img_CV_8U(uchar *img, uchar *imgout, int ImgWidth, int imgHeigh)
{
	int ImgNumColonne = blockIdx.x  * blockDim.x + threadIdx.x;
	int ImgNumLigne = blockIdx.y  * blockDim.y + threadIdx.y;
	int Index = (ImgNumLigne * ImgWidth + ImgNumColonne * 3);

	if ((ImgNumColonne < ImgWidth / 3) && (ImgNumLigne < imgHeigh))
	{
		/* Kernel Code Here */

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

		imgout[Index] = (uchar)(hue / 2);
		imgout[Index + 1] = (uchar)(saturation * 255);
		imgout[Index + 2] = (uchar)(value * 255);
	}

	return;
}

extern "C" bool GPGPU_TstImg_CV_8U(cv::Mat* img, cv::Mat* GPGPUimg)
{
	cudaError_t cudaStatus;
	uchar *devImage;
	uchar *devImageOut;

	unsigned int ImageSize = img->rows * img->step1();// step number of bytes in each row

													  // Allocate memory for image
	cudaStatus = cudaMalloc((void**)&devImage, ImageSize);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Upload the image to the GPU
	cudaStatus = cudaMemcpy(devImage, img->data, ImageSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 dimGrid(iDivUp(img->step1(), BLOCK_SIZE), iDivUp(img->cols, BLOCK_SIZE));
	dim3 dimGrid(iDivUp(img->cols, BLOCK_SIZE), iDivUp(img->rows, BLOCK_SIZE));


	// Test only
	// Allocate memory for the result image 
	cudaStatus = cudaMalloc((void**)&devImageOut, ImageSize);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	Kernel_Tst_Img_CV_8U << <dimGrid, dimBlock >> >(devImage, devImageOut, img->step1(), img->rows);
	// Check for any errors launching the kernel
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

	// Download the result image from gpu
	cudaStatus = cudaMemcpy(GPGPUimg->data, devImageOut, ImageSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(devImage);
	cudaFree(devImageOut);

	return cudaStatus;
}
// Transfert img to imgout to see how opencv image can be acces in GPGPU

//	TODO add color
__global__ 
void Kernel_ThresholdHSV(uchar *img, uchar *imgout, int ImgWidth, int imgHeigh, int minHue, int maxHue, int* backGroundColor, bool replaceForeground, int* ForegroundColor)
{
	int ImgNumColonne = blockIdx.x  * blockDim.x + threadIdx.x;
	int ImgNumLigne = blockIdx.y  * blockDim.y + threadIdx.y;
	int Index = (ImgNumLigne * ImgWidth) + (ImgNumColonne * 3);

	if ((ImgNumColonne < ImgWidth / 3) && (ImgNumLigne < imgHeigh))
	{
		int hue = img[Index];
		int saturation = img[Index + 1];
		int value = img[Index + 2];

		if (hue > minHue && hue < maxHue) {	//	Background-Green is black by default
			imgout[Index] = backGroundColor[0];
			imgout[Index + 1] = backGroundColor[1];
			imgout[Index + 2] = backGroundColor[2];
		}
		else {
			//	REPLACE WITH RGB COLORS
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
	Kernel_ThresholdHSV << <dimGrid, dimBlock >> >(gDevImage, gDevImageOut, imgHSV->step1(), imgHSV->rows, 38, 89, gBgColor, replaceForeground, gFgColor);	//	Green Hue range 38-98

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
int abs(int grad) {
	if (grad < 0) {
		return -1 * grad;
	}
	else {
		return grad;
	}
}

__global__ void Kernel_Sobel(uchar* img, uchar* imgout, int ImgWidth, int imgHeigh, int* gradientAll) // , int* maskX, int* maskY
{
	int ImgNumColonne = blockIdx.x  * blockDim.x + threadIdx.x;
	int ImgNumLigne = blockIdx.y  * blockDim.y + threadIdx.y;
	int Index = (ImgNumLigne * ImgWidth) + (ImgNumColonne * 3);


	int nani = (ImgNumLigne * (ImgWidth/3)) + ImgNumColonne;

	if ((ImgNumColonne < ImgWidth / 3) && (ImgNumLigne < imgHeigh)) 
	{

		if ((ImgNumColonne == ImgWidth - 1) || (ImgNumLigne == imgHeigh - 1) || (ImgNumColonne == ImgWidth - 2) || (ImgNumLigne == imgHeigh - 2)) {
			imgout[Index] = 0;
			imgout[Index + 1] = 0;
			imgout[Index + 2] = 0;

			gradientAll[nani] = 0;
		}
		else {
			int y = ImgNumLigne; // change imgnumligne pour y
			int x = ImgNumColonne;
			int i = Index;
			//imgout ->>> int 

			//	Gradient X ne pas calculer * 0
			int gradX = img[i] * -1 + img[i + 1] * 0 + img[i + 2] * 1;
			i = ((ImgNumLigne + 1) * ImgWidth) + (ImgNumColonne * 3);
			gradX += img[i] * -2 + img[i + 1] * 0 + img[i + 2] * 2;
			i = ((ImgNumLigne + 2) * ImgWidth) + (ImgNumColonne * 3);
			gradX += img[i] * -1 + img[i + 1] * 0 + img[i + 2] * 1;

			i = (ImgNumLigne * ImgWidth) + (ImgNumColonne * 3);

			//	Gradient Y
			int gradY = img[i] * -1 + img[i + 1] * -2 + img[i + 2] * -1;
			i = ((ImgNumLigne + 1) * ImgWidth) + (ImgNumColonne * 3);
			gradY += img[i] * 0 + img[i + 1] * 0 + img[i + 2] * 0;
			i = ((ImgNumLigne + 2) * ImgWidth) + (ImgNumColonne * 3);
			gradY += img[i] * 1 + img[i + 1] * 2 + img[i + 2] * 1;

			//	Gradient 
			int gradient = abs(gradX) + abs(gradY);
			int norm = gradient * 0.125;

			imgout[Index] = norm;
			imgout[Index + 1] = norm;
			imgout[Index + 2] = norm;

			
			gradientAll[nani] = norm;
		}	
	}

	return;
}

extern "C" bool GPGPU_Sobel(cv::Mat* imgTresh, cv::Mat* GPGPUimg, cv::Mat* Grayscale)
{
	//int maskX[9] = { -1, 0, 1,  -2, 0, 2,  -1, 0, 1 };
	//int maskY[9] = { -1, -2, -1,  0, 0, 0,  1, 2, 1 };

	//	1. Initialize data
	cudaError_t cudaStatus;
	uchar* gDevImage;
	uchar* gDevImageOut;
	int* gGradient;
	//int* gX;
	//int* gY;

	uint imageSize = imgTresh->rows * imgTresh->step1();
	uint gradientSize = imgTresh->rows * imgTresh->cols * sizeof(int);
	//uint maskSize = sizeof(maskX); //	Could be maskY

	/// Sobel
	int* gradient = new int[imgTresh->rows * imgTresh->cols];
	///

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(iDivUp(imgTresh->cols, BLOCK_SIZE), iDivUp(imgTresh->rows, BLOCK_SIZE));

	//	2. Allocation data
	cudaStatus = cudaMalloc(&gDevImage, imageSize);
	cudaStatus = cudaMalloc(&gDevImageOut, imageSize);
	cudaStatus = cudaMalloc(&gGradient, gradientSize);
	//cudaStatus = cudaMalloc(&gX, maskSize);
	//cudaStatus = cudaMalloc(&gY, maskSize);


	//	3. Copy data on GPU
	cudaStatus = cudaMemcpy(gDevImage, imgTresh->data, imageSize, cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy(gX, maskX, maskSize, cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy(gY, maskY, maskSize, cudaMemcpyHostToDevice);

	//	4. Launch kernel
	Kernel_Sobel << <dimGrid, dimBlock >> >(gDevImage, gDevImageOut, imgTresh->step1(), imgTresh->rows, gGradient); // , gX, gY

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
	cudaStatus = cudaMemcpy(gradient, gGradient, imageSize, cudaMemcpyDeviceToHost);

	//	6. Free GPU memory
Error:
	cudaFree(gDevImage);
	cudaFree(gDevImageOut);
	cudaFree(gGradient);

	return cudaStatus;
}
