#ifndef __MACRO_H__
#define __MACRO_H__

#include <iostream>
#include <string>
#include <cstdio>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define DEBUG
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

using time_point = decltype(std::chrono::steady_clock::now());

inline time_point get_time_point()
{
	CUDA_CHECK(cudaDeviceSynchronize());
	return std::chrono::steady_clock::now();
}

inline double get_duration(const time_point& from, const time_point& to)
{
	return std::chrono::duration_cast<std::chrono::duration<double>>(to - from).count();
}

#endif // !__MACRO_H__