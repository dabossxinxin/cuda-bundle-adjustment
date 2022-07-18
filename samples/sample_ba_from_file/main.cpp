#include <iostream>
#include <vector>
#include <chrono>

#include <opencv2/core.hpp>

#include <cuda_bundle_adjustment.h>
#include <object_creator.h>

static cuba::CudaBundleAdjustment::Ptr readGraph(const std::string& filename);

// use memory manager for vertices and edges, since CudaBundleAdjustment doesn't delete those pointers
static cuba::ObjectCreator obj;

int main(int argc, char** argv)
{
	/*if (argc < 2)
	{
		std::cout << "Usage: sample_ba_from_file input.json" << std::endl;
		return 0;
	}*/

	std::cout << "Reading Graph... " << std::flush;

	std::string filename =
		"D:\\Code\\cuda-bundle-adjustment\\samples\\ba_input\\ba_kitti_07.json";
	auto optimizer = readGraph(filename.c_str());

	std::cout << "Done." << std::endl << std::endl;

	std::cout << "=== Graph size : " << std::endl;
	std::cout << "num poses      : " << optimizer->nposes() << std::endl;
	std::cout << "num landmarks  : " << optimizer->nlandmarks() << std::endl;
	std::cout << "num edges      : " << optimizer->nedges() << std::endl << std::endl;

	std::cout << "Running BA... " << std::flush;

	const auto t0 = std::chrono::steady_clock::now();

	optimizer->initialize();
	optimizer->optimize(10);

	const auto t1 = std::chrono::steady_clock::now();

	std::cout << "Done." << std::endl << std::endl;

	const auto duration01 = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

	std::cout << "=== Processing time : " << std::endl;
	std::printf("BA total : %.2f[sec]\n\n", duration01);

	for (const auto& itTimeProfile : optimizer->timeProfile())
		std::printf("%-30s : %8.1f[msec]\n", itTimeProfile.first.c_str(), 1e3 * itTimeProfile.second);
	std::cout << std::endl;

	std::cout << "=== Objective function value : " << std::endl;
	for (const auto& stat : optimizer->batchStatistics())
		std::printf("iter: %2d, chi2: %.1f\n", stat.iteration + 1, stat.chi2);

	system("pause");
	return 0;
}

template <typename T, int N>
static inline cuba::Array<T, N> getArray(const cv::FileNode& node)
{
	cuba::Array<T, N> arr = {};
	int pos = 0;
	for (const auto& v : node)
	{
		arr[pos] = T(v);
		if (++pos >= N)
			break;
	}
	return arr;
}

static cuba::CudaBundleAdjustment::Ptr readGraph(const std::string& filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	CV_Assert(fs.isOpened());

	auto optimizer = cuba::CudaBundleAdjustment::create();

	// read pose vertices
	for (const auto& node : fs["pose_vertices"])
	{
		const int id = node["id"];
		const int fixed = node["fixed"];
		const auto q = Eigen::Quaterniond(getArray<double, 4>(node["q"]));
		const auto t = getArray<double, 3>(node["t"]);

		auto v = obj.create<cuba::PoseVertex>(id, q, t, fixed);
		optimizer->addPoseVertex(v);
	}

	// read landmark vertices
	for (const auto& node : fs["landmark_vertices"])
	{
		const int id = node["id"];
		const int fixed = node["fixed"];
		const auto Xw = getArray<double, 3>(node["Xw"]);

		auto v = obj.create<cuba::LandmarkVertex>(id, Xw, fixed);
		optimizer->addLandmarkVertex(v);
	}

	// read monocular edges
	for (const auto& node : fs["monocular_edges"])
	{
		const int iP = node["vertexP"];
		const int iL = node["vertexL"];
		const auto measurement = getArray<double, 2>(node["measurement"]);
		const double information = node["information"];

		auto vertexP = optimizer->poseVertex(iP);
		auto vertexL = optimizer->landmarkVertex(iL);

		auto e = obj.create<cuba::MonoEdge>(measurement, information, vertexP, vertexL);
		optimizer->addMonocularEdge(e);
	}

	// read stereo edges
	for (const auto& node : fs["stereo_edges"])
	{
		const int iP = node["vertexP"];
		const int iL = node["vertexL"];
		const auto measurement = getArray<double, 3>(node["measurement"]);
		const double information = node["information"];

		auto vertexP = optimizer->poseVertex(iP);
		auto vertexL = optimizer->landmarkVertex(iL);

		auto e = obj.create<cuba::StereoEdge>(measurement, information, vertexP, vertexL);
		optimizer->addStereoEdge(e);
	}

	// read camera parameters
	cuba::CameraParams camera;
	camera.fx = fs["fx"];
	camera.fy = fs["fy"];
	camera.cx = fs["cx"];
	camera.cy = fs["cy"];
	camera.bf = fs["bf"];

	optimizer->setCameraPrams(camera);

	// "warm-up" to avoid overhead
	optimizer->initialize();
	optimizer->optimize(1);

	return optimizer;
}