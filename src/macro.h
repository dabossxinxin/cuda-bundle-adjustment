#ifndef __MACRO_H__
#define __MACRO_H__

#include <iostream>
#include <string>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#define CUDA_CHECK(err) \
do {\
	if (err != cudaSuccess) { \
		printf("[CUDA Error] %s (code: %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
	} \
} while (0)

inline void SparseMatrixRepresentation(
	const int& size,
	const int& nnz,
	const int* rowPtr,
	const int* colInd,
	const std::string& filename
)
{
	cv::Mat sparse_matrix = cv::Mat(size, size, CV_8UC1, 255);
	for (int iti = 0; iti < size; ++iti) {
		int start = rowPtr[iti];
		int end = rowPtr[iti + 1];
		for (int itj = start; itj < end; ++itj) {
			sparse_matrix.at<uchar>(iti, colInd[itj]) = 0;
		}
	}
	cv::imwrite(filename.c_str(), sparse_matrix);
}

#endif // !__MACRO_H__