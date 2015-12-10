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
	//capt.open("data\\cam1\\background.avi");
	//cv::Mat a;
	//for (int i = 0; i < 50; i++)
	//	capt.read(a);
	//cv::imwrite("data\\cam1\\background.png", a);

	//CALIBRATION EXAMPLE
	//CameraCalibrator c("data\\cam1\\intrinsics.avi", cv::Size(644, 486), cv::Size(8, 6));
	//c.Calibrate();
	//c.SaveCalibration(std::string("data\\cam1\\intrinsics.xml"));
	//CameraIntrinsic in;
	//c.GetIntrinsic(in);
	//cv::VideoCapture vc("data\\cam1\\intrinsics.avi");
	//cv::Mat m, n;
	//for (int i = 0; i < 50; ++i)
	//	vc.read(m);
	//double a = 0;
	//double inc = 0.1;
	//while (true) {
	//	a += inc;
	//	if (a < 0 || a > 1)
	//	{
	//		inc *= -1;
	//		a += inc;
	//	}
	//	cv::copyMakeBorder(m, n, 50, 50, 50, 50, IPL_BORDER_CONSTANT, 0);
	//	
	//	cv::undistort(n.clone(), n, in.cameraMatrix, in.distCoeffs);
	//	cv::namedWindow("Window");
	//	cv::imshow("Window", n);
	//	if (cv::waitKey(30) == 'y') break;
	//}

	return EXIT_SUCCESS;
}
