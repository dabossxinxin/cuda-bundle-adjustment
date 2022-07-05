#ifndef __SCALAR_H__
#define __SCALAR_H__

#include <type_traits>

namespace cuba
{
#ifdef USE_FLOAT32
	using Scalar = float;
#else
	using Scalar = double;
#endif // USE_FLOAT32

	static_assert(std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value,
		"Scalar must be float or double.");
} // namespace cuba
#endif // !__SCALAR_H__
