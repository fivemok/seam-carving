#include "seam_kernels.cuh"

using namespace cv;
using namespace cv::cuda;


__global__
void cutSeamH(const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst, int *seam)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < dst.cols) {
		int offset = 0;
		int seamRow = seam[tid];

		for (int row = 0; row < dst.rows; row++) {
			if (row == seamRow) {
				offset = 1;
			}
			dst(row, tid) = src(row + offset, tid);
		}
	}

}

__global__
void cutSeamV(const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst, int *seam)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < dst.rows) {
		int offset = 0;
		int seamCol = seam[tid];

		for (int col = 0; col < dst.cols; col++) {
			if (col == seamCol) {
				offset = 1;
			}
			dst(tid, col) = src(tid, col + offset);
		}
	}

}
