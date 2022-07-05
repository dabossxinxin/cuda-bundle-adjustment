﻿#ifndef __CUDA_BLOCK_SOLVER_H__
#define __CUDA_BLOCK_SOLVER_H__

#include "device_matrix.h"

namespace cuba
{
	namespace gpu
	{
		void waitForKernelCompletion();

		void setCameraParameters(const Scalar* camera);

		void buildHplStructure(GpuVec3i& blockpos, GpuHplBlockMat& Hpl, GpuVec1i& indexPL, GpuVec1i& nnzPerCol);

		void findHschureMulBlockIndices(const GpuHplBlockMat& Hpl, const GpuHscBlockMat& Hsc,
			GpuVec3i& mulBlockIds);

		Scalar computeActiveErrors(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec3d& Xws,
			const GpuVec2d& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
			GpuVec2d& errors, GpuVec3d& Xcs, Scalar* chi);

		Scalar computeActiveErrors(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec3d& Xws,
			const GpuVec3d& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
			GpuVec3d& errors, GpuVec3d& Xcs, Scalar* chi);

		void constructQuadraticForm(const GpuVec3d& Xcs, const GpuVec4d& qs, const GpuVec2d& errors,
			const GpuVec1d& omegas, const GpuVec2i& edge2PL, const GpuVec1i& edge2Hpl, const GpuVec1b& flags,
			GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl);

		void constructQuadraticForm(const GpuVec3d& Xcs, const GpuVec4d& qs, const GpuVec3d& errors,
			const GpuVec1d& omegas, const GpuVec2i& edge2PL, const GpuVec1i& edge2Hpl, const GpuVec1b& flags,
			GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl);

		Scalar maxDiagonal(const GpuPxPBlockVec& Hpp, Scalar* maxD);

		Scalar maxDiagonal(const GpuLxLBlockVec& Hll, Scalar* maxD);

		void addLambda(GpuPxPBlockVec& Hpp, Scalar lambda, GpuPx1BlockVec& backup);

		void addLambda(GpuLxLBlockVec& Hll, Scalar lambda, GpuLx1BlockVec& backup);

		void restoreDiagonal(GpuPxPBlockVec& Hpp, const GpuPx1BlockVec& backup);

		void restoreDiagonal(GpuLxLBlockVec& Hll, const GpuLx1BlockVec& backup);

		void computeBschure(const GpuPx1BlockVec& bp, const GpuHplBlockMat& Hpl, const GpuLxLBlockVec& Hll,
			const GpuLx1BlockVec& bl, GpuPx1BlockVec& bsc, GpuLxLBlockVec& invHll, GpuPxLBlockVec& Hpl_invHll);

		void computeHschure(const GpuPxPBlockVec& Hpp, const GpuPxLBlockVec& Hpl_invHll,
			const GpuHplBlockMat& Hpl, const GpuVec3i& mulBlockIds, GpuHscBlockMat& Hsc);

		void convertHschureBSRToCSR(const GpuHscBlockMat& HscBSR, const GpuVec1i& BSR2CSR, GpuVec1d& HscCSR);

		void twistCSR(int size, int nnz, const int* srcRowPtr, const int* srcColInd, const int* P,
			int* dstRowPtr, int* dstColInd, int* dstMap, int* nnzPerRow);

		void permute(int size, const Scalar* src, Scalar* dst, const int* P);

		void schurComplementPost(const GpuLxLBlockVec& invHll, const GpuLx1BlockVec& bl,
			const GpuHplBlockMat& Hpl, const GpuPx1BlockVec& xp, GpuLx1BlockVec& xl);

		void updatePoses(const GpuPx1BlockVec& xp, GpuVec4d& qs, GpuVec3d& ts);

		void updateLandmarks(const GpuLx1BlockVec& xl, GpuVec3d& Xws);

		void computeScale(const GpuVec1d& x, const GpuVec1d& b, Scalar* scale, Scalar lambda);

	} // namespace gpu

} // namespace cuba

#endif // !__CUDA_BLOCK_SOLVER_H__
