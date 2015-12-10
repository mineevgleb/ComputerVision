#include <cstdlib>
#include <string>

#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include "CameraCalibrator.h"
#include <vector>

using namespace std;
using namespace nl_uu_science_gmt;

int main(
		int argc, char** argv)
{
	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv);

	//cv::VideoCapture capt;
	//capt.open("data\\cam4\\background.avi");
	//cv::Mat a;
	//for (int i = 0; i < 50; i++)
	//	capt.read(a);
	//cv::imwrite("data\\cam4\\background.png", a);

	//CALIBRATION EXAMPLE
	//CameraCalibrator c("data\\cam3\\intrinsics.avi", cv::Size(644, 486), cv::Size(8, 6));
	//c.Calibrate();
	//c.SaveCalibration(std::string("data\\cam3\\intrinsics.xml"));
	//CameraIntrinsic in;
	//c.GetIntrinsic(in);
	//cv::VideoCapture vc("data\\cam3\\intrinsics.avi");
	//cv::Mat m, n;
	//for (int i = 0; i < 50; ++i)
	//vc.read(m);
	//n = m.clone();
	//cv::undistort(n, m, in.cameraMatrix, in.distCoeffs);
	//cv::namedWindow("Window");
	//cv::imshow("Window", m);
	//cv::waitKey();

	return EXIT_SUCCESS;
}
