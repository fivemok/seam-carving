#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

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

	if (tid < energy.rows) {
		for (int col = 0; col < energy.cols; col++) {
			T val = sqrt(pow(gradY(tid, col), 2) + pow(gradX(tid, col), 2));
			energy(tid, col) = val;
		}
	}
}

template <typename T> __global__
void computeEnergyMapH(const cv::cuda::PtrStepSz<T> src,
	cv::cuda::PtrStepSz<T> dst, int col)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < src.rows) {
		if (col == 0) {
			dst(tid, col) = src(tid, col);
		}
		else {
			if (tid == 0) {
				dst(tid, col) = src(tid, col)
					+ MIN(dst(tid, col - 1), dst(tid + 1, col - 1));
			}
			else if (tid == src.rows - 1) {
				dst(tid, col) = src(tid, col)
					+ MIN(dst(tid, col - 1), dst(tid - 1, col - 1));
			}
			else {
				dst(tid, col) = src(tid, col)
					+ MIN(dst(tid, col - 1), MIN(dst(tid - 1, col - 1), dst(tid + 1, col - 1)));
			}
		}
	}

}

template <typename T> __global__
void computeEnergyMapV(const cv::cuda::PtrStepSz<T> src,
	cv::cuda::PtrStepSz<T> dst, int row)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < src.cols) {
		if (row == 0) {
			dst(row, tid) = src(row, tid);
		}
		else {
			if (tid == 0) {
				dst(row, tid) = src(row, tid)
					+ MIN(dst(row - 1, tid), dst(row - 1, tid + 1));
			}
			else if (tid == src.cols - 1) {
				dst(row, tid) = src(row, tid)
					+ MIN(dst(row - 1, tid), dst(row - 1, tid - 1));
			}
			else {
				dst(row, tid) = src(row, tid)
					+ MIN(dst(row - 1, tid), MIN(dst(row - 1, tid - 1), dst(row - 1, tid + 1)));
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
