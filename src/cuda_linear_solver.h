#ifndef __CUDA_LINEAR_SOLVER_H__
#define __CUDA_LINEAR_SOLVER_H__

#include <memory>

#include "scalar.h"
#include "sparse_block_matrix.h"

namespace cuba
{

	class SparseLinearSolver
	{
	public:

		using Ptr = std::unique_ptr<SparseLinearSolver>;
		static Ptr create();

		virtual void initialize(const HschurSparseBlockMatrix& Hsc) = 0;
		virtual bool solve(const Scalar* d_A, const Scalar* d_b, Scalar* d_x) = 0;

		virtual ~SparseLinearSolver();
	};

} // namespace cuba

#endif // !__CUDA_LINEAR_SOLVER_H__
