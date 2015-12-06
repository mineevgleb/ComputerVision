#include <cstdlib>
#include <string>

#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include "CameraCalibrator.h"
#include <vector>

using namespace std;
using namespace nl_uu_science_gmt;

double deltaE2000(const cv::Vec3f &col1, const cv::Vec3f &col2) {
	double L1 = ((double)col1[0]);
	double L2 = ((double)col2[0]);
	double a1 = col1[1];
	double a2 = col2[1];
	double b1 = col1[2];
	double b2 = col2[2];
	double dL = L1 - L2;
	double C1 = sqrt(a1 * a1 + b1 * b1);
	double C2 = sqrt(a2 * a2 + b2 * b2);
	double dC = C1 - C2;
	double da = a1 - a2;
	double db = b1 - b2;
	double dH = sqrt(da * da + db * db - dC * dC);
	double SC = 1.0 + 0.045 * C1;
	double SH = 1.0 + 0.015 * C1;
	return sqrt(pow(dL, 2) + pow(dC / SC, 2) + pow(dH / SH, 2));
}

double deltaTone(const cv::Vec3f &col1, const cv::Vec3f &col2) {
	double a1 = col1[1];
	double a2 = col2[1];
	double b1 = col1[2];
	double b2 = col2[2];
	double C1 = sqrt(a1 * a1 + b1 * b1);
	double C2 = sqrt(a2 * a2 + b2 * b2);
	double dC = C1 - C2;
	double da = a1 - a2;
	double db = b1 - b2;
	double dH = sqrt(da * da + db * db - dC * dC);
	double SC = 1.0 + 0.045 * C1;
	double SH = 1.0 + 0.015 * C1;
	return sqrt(pow(dC / SC, 2) + pow(dH / SH, 2));
}

int main(
		int argc, char** argv)
{
	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv);

	//cv::Mat back = cv::imread("data\\cam1\\background.png");
	//cv::Mat backFloat;
	//back.convertTo(backFloat, CV_32F);
	//backFloat /= 255;
	//cv::VideoCapture capt;
	//capt.open("data\\cam1\\video.avi");
	//cv::Mat frame;
	//capt.read(frame);
	//cv::Mat frameFloat;
	//frame.convertTo(frameFloat, CV_32F);
	//frameFloat /= 255;
	//cv::Mat backLab;
	//cv::Mat frameLab;
	//cv::cvtColor(backFloat, backLab, CV_BGR2Lab);
	//cv::cvtColor(frameFloat, frameLab, CV_BGR2Lab);
	//cv::Mat diff(backLab.rows, backLab.cols, CV_32F);
	//for (int i = 0; i < backLab.rows; ++i) {
	//	for (int j = 0; j < backLab.cols; ++j) {
	//		cv::Vec3f framePx = frameLab.at<cv::Vec3f>(cv::Point(j, i));
	//		cv::Vec3f backPx = backLab.at<cv::Vec3f>(cv::Point(j, i));
	//		double delta = deltaE2000(framePx, backPx);
	//		double toneDist = deltaTone(framePx, backPx);
	//		if (framePx[0] < backPx[0]) {
	//			delta *= pow(toneDist / 100, 0.5);
	//		}
	//		diff.at<float>(cv::Point(j, i)) = delta / 100;
	//
	//	}
	//}
	//diff *= 255;
	//cv::Mat diffBit;
	//diff.convertTo(diffBit, CV_8UC1);
	//cv::Mat tmp;
	//vector<vector<cv::Point> > contours;
	//vector<cv::Vec4i> hierarchy;
	////cv::findContours(diffBit, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1);
	////cv::drawContours(diffBit, contours, -1, cv::Scalar(0, 0, 255), 2);
	//
	//cv::namedWindow("Window");
	//cv::imshow("Window", diffBit);
	////cv::namedWindow("Window2");
	////cv::imshow("Window2", tmp2);
//	//cv::namedWindow("Window3");
//	//cv::imshow("Window3", tmp);
	//cv::waitKey();

	//BASIC BACKGROUND EXTRACTION. NEED TO BE IMPROVED
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
