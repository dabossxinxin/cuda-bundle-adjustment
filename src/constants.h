#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

namespace cuba
{

static constexpr int PDIM = 6;
static constexpr int LDIM = 3;

enum StorageOrder
{
	ROW_MAJOR,
	COL_MAJOR
};

enum EdgeFlag
{
	EDGE_FLAG_FIXED_L = 1,
	EDGE_FLAG_FIXED_P = 2
};

} // namespace cuba

#endif // !__CONSTANTS_H__
