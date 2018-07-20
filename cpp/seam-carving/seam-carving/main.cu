#include <iostream>

#include "seam.cuh"
#include "seam_kernels.cuh"

using namespace cv;
using namespace std;

int main()
{
	Mat h_image = imread("C:/Users/mknic/Desktop/git/seam-carving/images/camel.jpg");
	//Mat h_image = imread("C:/Users/mknic/Desktop/git/seam-carving/images/castle.jpg");
	//Mat h_image = imread("C:/Users/mknic/Desktop/git/seam-carving/images/toronto.jpg");
	cuda::setDevice(0);
	if (h_image.empty()) {
		cout << "Transferring host image to device failed" << endl;
		return -1;
	}

	Mat h_resizedImage = cutSeams<double>(h_image, 250, HORIZONTAL);
	imshow("Original", h_image);
	imshow("Resized", h_resizedImage);
	imwrite("C:/Users/mknic/Desktop/HOHO.PNG", h_resizedImage);
	waitKey(0);

	h_image.release();
	h_resizedImage.release();

	return 0;
}
