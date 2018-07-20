#pragma once

#include <math.h>
#include "seam.cuh"
#include "seam_kernels.cuh"

//#define DEBUG
#define DEPTH_ENERGY	CV_64F
#define DEPTH_SOBEL		CV_16S
#define DEPTH_DEBUG		CV_8UC1
#define SIGMA			0


static inline void computeGradient(cv::Ptr<cv::cuda::Filter> gauss,
	cv::Ptr<cv::cuda::Filter> sobelX, cv::Ptr<cv::cuda::Filter> sobelY,
	cv::cuda::GpuMat& d_image, cv::cuda::GpuMat& d_grayscale,
	cv::cuda::GpuMat& d_gradX, cv::cuda::GpuMat& d_gradY)
{
	gauss->apply(d_image, d_grayscale);
	cv::cuda::cvtColor(d_grayscale, d_grayscale, cv::COLOR_BGR2GRAY);
	cv::cuda::GpuMat d_gradYTemp, d_gradXTemp;
	sobelY->apply(d_grayscale, d_gradYTemp);
	sobelX->apply(d_grayscale, d_gradXTemp);
	d_gradYTemp.convertTo(d_gradY, DEPTH_ENERGY);
	d_gradXTemp.convertTo(d_gradX, DEPTH_ENERGY);
}

template <typename T>
static inline void computeEnergyMap(const cv::cuda::PtrStepSz<T> d_grad,
	cv::cuda::PtrStepSz<T> d_energy, bool isHorizontal)
{
	int dim = isHorizontal ? d_grad.cols : d_grad.rows;
	int numThreads = isHorizontal ? d_grad.rows : d_grad.cols;
	int blocks = ceil(((double)numThreads)/MAX_THRDS);

	for (int i = 0; i < dim; i++) {
		if (isHorizontal) {
			computeEnergyMapH<T><<<blocks,MAX_THRDS>>>(d_grad, d_energy, i);
		}
		else {
			computeEnergyMapV<T><<<blocks,MAX_THRDS>>>(d_grad, d_energy, i);
		}
	}
}

template <typename T>
cv::Mat cutSeams(cv::Mat h_image, int numSeams, Orientation o)
{
	cv::Mat h_resized, h_energy, h_grad;
	cv::cuda::GpuMat d_image(h_image), d_grayscale, d_energy;
	cv::cuda::GpuMat d_grad, d_gradX, d_gradY;
	cv::cuda::cvtColor(d_image, d_grayscale, cv::COLOR_BGR2GRAY);

	const cv::Ptr<cv::cuda::Filter> gauss =
		cv::cuda::createGaussianFilter(d_image.type(), d_image.type(), cv::Size(3, 3), SIGMA);
	const cv::Ptr<cv::cuda::Filter> sobelX =
		cv::cuda::createSobelFilter(d_grayscale.type(), DEPTH_SOBEL, 1, 0);
	const cv::Ptr<cv::cuda::Filter> sobelY =
		cv::cuda::createSobelFilter(d_grayscale.type(), DEPTH_SOBEL, 0, 1);

	bool isHorizontal = (o == HORIZONTAL);
	int seamDim = isHorizontal ? d_image.cols : d_image.rows;
	int *seam = NULL;
	cudaMalloc((void**)&seam, sizeof(int)*seamDim);

	int newRows = d_image.rows;
	int newCols = d_image.cols;

	for (int i = 0; i < numSeams; i++) {
		computeGradient(gauss, sobelX, sobelY, d_image, d_grayscale, d_gradX, d_gradY);
		d_energy = cv::cuda::GpuMat(d_gradY.rows, d_gradY.cols, d_gradY.type());
		d_grad = cv::cuda::GpuMat(d_gradY.rows, d_gradY.cols, d_gradY.type());
		cv::cuda::PtrStepSz<T> d_gradYPtr(d_gradY.rows, d_gradY.cols, d_gradY.ptr<T>(), d_gradY.step);
		cv::cuda::PtrStepSz<T> d_gradXPtr(d_gradX.rows, d_gradX.cols, d_gradX.ptr<T>(), d_gradX.step);
		cv::cuda::PtrStepSz<T> d_gradPtr(d_grad.rows, d_grad.cols, d_grad.ptr<T>(), d_grad.step);
		cv::cuda::PtrStepSz<T> d_energyPtr(d_energy.rows, d_energy.cols, d_energy.ptr<T>(), d_energy.step);
		int blocks = ceil(((double)newRows)/MAX_THRDS);
		computeGrad<T><<<blocks,MAX_THRDS>>>(d_gradY, d_gradX, d_grad);
		computeEnergyMap<T>(d_gradPtr, d_energyPtr, isHorizontal);

		if (isHorizontal) {
			findSeamH<T><<<1,1>>>(d_energyPtr, seam);
			newRows -= 1;
		}
		else {
			findSeamV<T><<<1,1>>>(d_energyPtr, seam);
			newCols -= 1;
		}

		cv::cuda::GpuMat d_imageTemp(newRows, newCols, d_image.type());
		cv::cuda::PtrStepSz<uchar3> d_imagePtr(d_image.rows, d_image.cols, d_image.ptr<uchar3>(), d_image.step);
		cv::cuda::PtrStepSz<uchar3> d_imageTempPtr(newRows, newCols, d_imageTemp.ptr<uchar3>(), d_imageTemp.step);
		
		if (isHorizontal) {
			int blocks = ceil(((double)newCols)/MAX_THRDS);
			cutSeamH<<<blocks,MAX_THRDS>>>(d_imagePtr, d_imageTempPtr, seam);
		}
		else {
			int blocks = ceil(((double)newRows)/MAX_THRDS);
			cutSeamV<<<blocks,MAX_THRDS>>>(d_imagePtr, d_imageTempPtr, seam);
		}

		d_image = cv::cuda::GpuMat(d_imageTemp);
		d_imageTemp.release();

#ifdef DEBUG
		cv::cuda::GpuMat d_energyTemp, d_gradTemp;
		cv::Mat h_energy, h_grad;
		d_energy.convertTo(d_energyTemp, DEPTH_DEBUG);
		d_energyTemp.download(h_energy);
		d_grad.convertTo(d_gradTemp, DEPTH_DEBUG);
		d_gradTemp.download(h_grad);
		d_image.download(h_resized);
		cv::imshow("DEBUG: resized", h_resized);
		cv::imshow("DEBUG: energy", h_energy);
		cv::imshow("DEBUG: grad", h_grad);
		cv::waitKey(0);
		d_energyTemp.release(); d_gradTemp.release();
		h_energy.release(); h_grad.release();
#endif
	}

	d_image.download(h_resized);
	d_image.release();
	d_grayscale.release();
	d_energy.release();
	d_grad.release();
	d_gradX.release();
	d_gradY.release();
	h_energy.release();
	h_grad.release();
	cudaFree(seam);

	return h_resized;
}
