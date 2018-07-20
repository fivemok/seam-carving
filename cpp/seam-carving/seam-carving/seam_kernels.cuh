#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

//#define DEBUG_SEAM
#define TPB 256

enum Orientation { HORIZONTAL, VERTICAL };

__global__
void cutSeam(const cv::cuda::PtrStepSz<uchar3> src,
	cv::cuda::PtrStepSz<uchar3> dst, int *seam, Orientation o)
{
	bool isHorizontal = (o == HORIZONTAL);
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int tLimit = isHorizontal ? dst.cols : dst.rows;

	if (tid < tLimit) {
		int dim = isHorizontal ? dst.rows : dst.cols;
		int offset = 0;
		int seamLoc = seam[tid];

		for (int rc = 0; rc < dim; rc++) {
			if (rc == seamLoc) {
				offset = 1;
			}

			if (isHorizontal) {
				dst(rc, tid) = src(rc + offset, tid);
			}
			else {
				dst(tid, rc) = src(tid, rc + offset);
			}
		}

	}
}

template <typename T> __global__
void computeEnergy(const cv::cuda::PtrStepSz<T> gradY,
	const cv::cuda::PtrStepSz<T> gradX, cv::cuda::PtrStepSz<T> energy)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int tLimit = energy.rows * energy.cols;

	if (tid < tLimit) {
		for (int i = tid; i < tLimit; i += gridDim.x * blockDim.x) {
			int y = i / energy.cols;
			int x = i - y * energy.cols;
			T val = sqrt(pow(gradY(y, x), 2) + pow(gradX(y, x), 2));
			energy(y, x) = val;
		}
	}
}

template <typename T> __global__
void computeEnergyMap(const cv::cuda::PtrStepSz<T> src,
	cv::cuda::PtrStepSz<T> dst, int rc, Orientation o)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	bool isHorizontal = (o == HORIZONTAL);
	int tLimit = isHorizontal ? dst.rows : dst.cols;

	if (tid < tLimit) {
		if (rc == 0) {
			if (isHorizontal) {
				dst(tid, rc) = src(tid, rc);
			}
			else {
				dst(rc, tid) = src(rc, tid);
			}
		}
		else {
			int diagOneIdx = MAX(tid - 1, 0);
			int diagTwoIdx = MIN(tid + 1, tLimit - 1);
			if (isHorizontal) {
				T diagOne = dst(diagOneIdx, rc - 1);
				T diagTwo = dst(diagTwoIdx, rc - 1);
				T adjacent = dst(tid, rc - 1);
				dst(tid, rc) = src(tid, rc) + MIN(adjacent, MIN(diagOne, diagTwo));
			}
			else {
				T diagOne = dst(rc - 1, diagOneIdx);
				T diagTwo = dst(rc - 1, diagTwoIdx);
				T adjacent = dst(rc - 1, tid);
				dst(rc, tid) = src(rc, tid) + MIN(adjacent, MIN(diagOne, diagTwo));
			}
		}
	}
}

template <typename T> __global__
void findSeam(const cv::cuda::PtrStepSz<T> src, int *seam, Orientation o)
{
	bool isHorizontal = (o == HORIZONTAL);
	T currMin = INFINITY;
	int minRc = 0;
	int lastRc = isHorizontal ? src.cols - 1 : src.rows - 1;
	int dim = isHorizontal ? src.rows : src.cols;
	for (int rc = 0; rc < dim; rc++) {
		T currEnergy;
		if (isHorizontal) {
			currEnergy = src(rc, lastRc);
		}
		else {
			currEnergy = src(lastRc, rc);
		}

		if (currEnergy < currMin) {
			currMin = currEnergy;
			minRc = rc;
		}
	}

	seam[lastRc] = minRc;

#ifdef DEBUG_SEAM
	printf("MIN RC: %d @ %f\n", minRc, currMin);
#endif

	for (int rc = lastRc - 1; rc >= 0; rc--) {
		int diagOneIdx = MAX(minRc - 1, 0);
		int diagTwoIdx = MIN(minRc + 1, dim - 1);
		T diagOne, diagTwo, adjacent;
		if (isHorizontal) {
			diagOne = src(diagOneIdx, rc);
			diagTwo = src(diagTwoIdx, rc);
			adjacent = src(minRc, rc);
		}
		else {
			diagOne = src(rc, diagOneIdx);
			diagTwo = src(rc, diagTwoIdx);
			adjacent = src(rc, minRc);
		}

		T minVal = MIN(adjacent, MIN(diagOne, diagTwo));
		if (minVal != adjacent) {
			if (minVal == diagOne) {
				minRc = diagOneIdx;
			}
			else {
				minRc = diagTwoIdx;
			}
		}

		seam[rc] = minRc;
	}

}
