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
void computeGrad(const cv::cuda::PtrStepSz<T> gradX,
	const cv::cuda::PtrStepSz<T> gradY, cv::cuda::PtrStepSz<T> energy)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < energy.rows) {
		for (int col = 0; col < energy.cols; col++) {
			T val = sqrt(pow(gradX(tid, col), 2) + pow(gradY(tid, col), 2));
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
void findSeamH(const cv::cuda::PtrStepSz<T> src, int *seam)
{
	T currMin = INFINITY;
	int minRow = 0;
	int lastCol = src.cols - 1;
	for (int row = 0; row < src.rows; row++) {
		T currEnergy = src(row, lastCol);
		if (currEnergy < currMin) {
			currMin = currEnergy;
			minRow = row;
		}
	}

	seam[src.cols - 1] = minRow;

#ifdef DEBUG_SEAM
	printf("MIN ROW: %d @ %f\n", minRow, currMin);
#endif

	for (int col = src.cols - 2; col >= 0; col--) {
		int upperLeftRowIdx = MAX(minRow - 1, 0);
		int lowerLeftRowIdx = MIN(minRow + 1, src.rows - 1);
		T  upperLeft = src(upperLeftRowIdx, col);
		T		left = src(minRow, col);
		T  lowerLeft = src(lowerLeftRowIdx, col);
		T	  minVal = MIN(left, MIN(upperLeft, lowerLeft));
		if (minVal != left) {
			if (minVal == upperLeft) {
				minRow = upperLeftRowIdx;
			}
			else {
				minRow = lowerLeftRowIdx;
			}
		}

		seam[col] = minRow;
	}

}

template <typename T> __global__
void findSeamV(const cv::cuda::PtrStepSz<T> src, int *seam)
{
	T currMin = INFINITY;
	int minCol = 0;
	int lastRow = src.rows - 1;
	for (int col = 0; col < src.cols; col++) {
		T currEnergy = src(lastRow, col);
		if (currEnergy < currMin) {
			currMin = currEnergy;
			minCol = col;
		}
	}

	seam[src.rows - 1] = minCol;

#ifdef DEBUG_SEAM
	printf("MIN COL %d @ %f\n", minCol, currMin);
#endif

	for (int row = src.rows - 2; row >= 0; row--) {
		int  upperLeftColIdx = MAX(minCol - 1, 0);
		int upperRightColIdx = MIN(minCol + 1, src.cols - 1);
		T  upperLeft = src(row, upperLeftColIdx);
		T		  up = src(row, minCol);
		T upperRight = src(row, upperRightColIdx);
		T	  minVal = MIN(up, MIN(upperLeft, upperRight));
		if (minVal != up) {
			if (minVal == upperLeft) {
				minCol = upperLeftColIdx;
			}
			else {
				minCol = upperRightColIdx;
			}
		}
		
		seam[row] = minCol;
	}
}
