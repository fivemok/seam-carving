#pragma once

#include <math.h>
#include "seam.cuh"
#include "seam_kernels.cuh"

//#define DEBUG
#define DEPTH_ENERGY	CV_64F
#define DEPTH_SOBEL		CV_16S
#define DEPTH_DEBUG		CV_8UC1
#define SIGMA			0

template <typename T>
static inline void computeEnergy(cv::Ptr<cv::cuda::Filter> gauss,
	cv::Ptr<cv::cuda::Filter> sobelY, cv::Ptr<cv::cuda::Filter> sobelX,
	cv::cuda::GpuMat &d_image, cv::cuda::GpuMat &d_grayscale,
	cv::cuda::GpuMat &d_gradY, cv::cuda::GpuMat &d_gradX, cv::cuda::GpuMat &d_energy)
{
	gauss->apply(d_image, d_grayscale);
	cv::cuda::cvtColor(d_grayscale, d_grayscale, cv::COLOR_BGR2GRAY);
	cv::cuda::GpuMat d_gradYTemp, d_gradXTemp;
	sobelY->apply(d_grayscale, d_gradYTemp);
	sobelX->apply(d_grayscale, d_gradXTemp);
	d_gradYTemp.convertTo(d_gradY, DEPTH_ENERGY);
	d_gradXTemp.convertTo(d_gradX, DEPTH_ENERGY);
	d_energy = cv::cuda::GpuMat(d_gradY.rows, d_gradY.cols, d_gradY.type());
	int blocks = ceil(((double)d_energy.rows)/TPB);
	computeEnergy<T><<<blocks,TPB>>>(d_gradY, d_gradX, d_energy);
}

template <typename T>
static inline void computeEnergyMap(const cv::cuda::PtrStepSz<T> d_energy,
	cv::cuda::PtrStepSz<T> d_energyMap, Orientation o)
{
	bool isHorizontal = (o == HORIZONTAL);
	int dim = isHorizontal ? d_energy.cols : d_energy.rows;
	int numThreads = isHorizontal ? d_energy.rows : d_energy.cols;
	int blocks = ceil(((double)numThreads)/TPB);

	for (int i = 0; i < dim; i++) {
		computeEnergyMap<T><<<blocks,TPB>>>(d_energy, d_energyMap, i, o);
	}
}

template <typename T>
cv::Mat cutSeams(cv::Mat h_image, int numSeams, Orientation o)
{
	cv::Mat h_resized;
	cv::cuda::GpuMat d_image(h_image);
	cv::cuda::GpuMat d_grayscale, d_gradY, d_gradX, d_energy, d_energyMap;
	cv::cuda::cvtColor(d_image, d_grayscale, cv::COLOR_BGR2GRAY);

	const cv::Ptr<cv::cuda::Filter> gauss =
		cv::cuda::createGaussianFilter(d_image.type(), d_image.type(), cv::Size(3, 3), SIGMA);
	const cv::Ptr<cv::cuda::Filter> sobelX =
		cv::cuda::createSobelFilter(d_grayscale.type(), DEPTH_SOBEL, 1, 0);
	const cv::Ptr<cv::cuda::Filter> sobelY =
		cv::cuda::createSobelFilter(d_grayscale.type(), DEPTH_SOBEL, 0, 1);

	bool isHorizontal = (o == HORIZONTAL);
	int newRows = d_image.rows;
	int newCols = d_image.cols;
	int &newDim = isHorizontal ? newRows : newCols;
	int &cutSeamTs = isHorizontal ? newCols : newRows;
	int *seam = NULL;
	int seamDim = isHorizontal ? d_image.cols : d_image.rows;
	cudaMalloc((void**)&seam, sizeof(int)*seamDim);

	for (int i = 0; i < numSeams; i++) {
		computeEnergy<T>(gauss, sobelY, sobelX, d_image, d_grayscale, d_gradY, d_gradX, d_energy);
		cv::cuda::GpuMat d_energyMap(d_energy.rows, d_energy.cols, d_energy.type());
		cv::cuda::PtrStepSz<T> d_gradYPtr(d_gradY.rows, d_gradY.cols, d_gradY.ptr<T>(), d_gradY.step);
		cv::cuda::PtrStepSz<T> d_gradXPtr(d_gradX.rows, d_gradX.cols, d_gradX.ptr<T>(), d_gradX.step);
		cv::cuda::PtrStepSz<T> d_energyPtr(d_energy.rows, d_energy.cols, d_energy.ptr<T>(), d_energy.step);
		cv::cuda::PtrStepSz<T> d_energyMapPtr(d_energy.rows, d_energy.cols, d_energyMap.ptr<T>(), d_energy.step);

		computeEnergyMap<T>(d_energyPtr, d_energyMapPtr, o);
		findSeam<<<1,1>>>(d_energyMapPtr, seam, o);
		newDim--;

		cv::cuda::GpuMat d_imageTemp(newRows, newCols, d_image.type());
		cv::cuda::PtrStepSz<uchar3> d_imagePtr(d_image.rows, d_image.cols, d_image.ptr<uchar3>(), d_image.step);
		cv::cuda::PtrStepSz<uchar3> d_imageTempPtr(newRows, newCols, d_imageTemp.ptr<uchar3>(), d_imageTemp.step);
	
		int blocks = ceil(((double)cutSeamTs)/TPB);
		cutSeam<<<blocks,TPB>>>(d_imagePtr, d_imageTempPtr, seam, o);
		d_image = cv::cuda::GpuMat(d_imageTemp);
		d_imageTemp.release();

#ifdef DEBUG
		cv::cuda::GpuMat d_energyTemp, d_energyMapTemp;
		cv::Mat h_energy, h_energyMap;
		d_energy.convertTo(d_energyTemp, DEPTH_DEBUG);
		d_energyTemp.download(h_energy);
		d_energyMap.convertTo(d_energyMapTemp, DEPTH_DEBUG);
		d_energyMapTemp.download(h_energyMap);
		d_image.download(h_resized);
		cv::imshow("DEBUG: Resized", h_resized);
		cv::imshow("DEBUG: Energy", h_energy);
		cv::imshow("DEBUG: Energy Map", h_energyMap);
		cv::waitKey(0);
		d_energyTemp.release(); d_energyMapTemp.release();
		h_energy.release(); h_energyMap.release();
#endif
		d_energyMap.release();
	}

	d_image.download(h_resized);
	d_image.release();
	d_grayscale.release();
	d_energy.release();
	d_gradY.release();
	d_gradX.release();
	cudaFree(seam);

	return h_resized;
}
