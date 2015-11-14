#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include <strstream>
#include <vector>
#include <ctime>
#include "CameraThread.h"
#include "CheckboardThread.h"
#include "CameraCalibrator.h"

const clock_t CAPTURE_DELAY = CLOCKS_PER_SEC / 2;
const int FRAMES_TO_CALIBRATE = 12;

enum State {
	Calibration,
	Run,
	RunOriginal
};

void ShowMessage(cv::Mat &img, const char *msg, cv::Point pos) {
	cv::putText(img, msg, cv::Point(pos.x + 1, pos.y + 1), CV_FONT_HERSHEY_SIMPLEX,
		0.5, cv::Scalar(0, 0, 0, 255), 1);
	cv::putText(img, msg, pos, CV_FONT_HERSHEY_SIMPLEX,
		0.5, cv::Scalar(50, 50, 200, 255), 1, CV_AA);
}

cv::Point2f TranslatePointToScreen(const cv::Point3f &pt, const cv::Mat &translationMatrix) {
	cv::Mat ptMat = (cv::Mat_<double>(4, 1) << pt.x, pt.y, pt.z, 1);
	cv::Mat onscreenMat = translationMatrix * ptMat;
	return cv::Point2f(onscreenMat.at<double>(0) / onscreenMat.at<double>(2),
		onscreenMat.at<double>(1) / onscreenMat.at<double>(2));
}

void DrawCube(cv::Mat &frame, const cv::Mat & translationMatrix, const cv::Point3f &origin, double edgeSize) 
{
	cv::Point3f cube[8];
	for (int i = 0; i < 8; i++)
		cube[i] = origin;
	cube[1].x += edgeSize;
	cube[2].x += edgeSize; cube[2].y += edgeSize;
	cube[3].y += edgeSize;
	cube[4].z -= edgeSize;
	cube[5].x += edgeSize; cube[5].z -= edgeSize;
	cube[6].x += edgeSize; cube[6].y += edgeSize; cube[6].z -= edgeSize;
	cube[7].y += edgeSize; cube[7].z -= edgeSize;
	cv::Point2f cubeOnScreen[8];
	for (int i = 0; i < 8; ++i)
		cubeOnScreen[i] = TranslatePointToScreen(cube[i], translationMatrix);
	for (int i = 0; i < 4; ++i) {
		cv::line(frame, cubeOnScreen[i], cubeOnScreen[(i + 1) % 4], cv::Scalar(0, 200, 255, 125), 2);
	}
	for (int i = 0; i < 4; ++i) {
		cv::line(frame, cubeOnScreen[i + 4], cubeOnScreen[(i + 1) % 4 + 4], cv::Scalar(0, 200, 255, 255), 2);
	}
	for (int i = 0; i < 4; ++i) {
		cv::line(frame, cubeOnScreen[i], cubeOnScreen[i + 4], cv::Scalar(0, 200, 255, 255), 2);
	}
}

void DrawAxis(cv::Mat &frame, const cv::Mat & translationMatrix, double len)
{
	cv::Point3f axis[4];
	for (int i = 0; i < 4; i++)
		axis[i] = cv::Point3f(0, 0, 0);
	axis[1].x = len;
	axis[2].y = len;
	axis[3].z = -len;
	cv::Point2f axisOnScreen[4];
	for (int i = 0; i < 4; ++i)
		axisOnScreen[i] = TranslatePointToScreen(axis[i], translationMatrix);
	cv::line(frame, axisOnScreen[0], axisOnScreen[1], cv::Scalar(0, 0, 255, 255), 2);
	cv::line(frame, axisOnScreen[0], axisOnScreen[2], cv::Scalar(0, 255, 0, 255), 2);
	cv::line(frame, axisOnScreen[0], axisOnScreen[3], cv::Scalar(255, 0, 0, 255), 2);
}

int main(int argc, char **argv) {
	CameraThread cam(0, true);
	CheckboardThread chk(&cam, true);
	CameraCalibrator calib(&chk, cam.GetFrame().size());
	calib.Calibrate();
	const char *windowName = "Calibration";
	cv::namedWindow(windowName);
	bool undistortionActive = true;
	bool isMirrored = true;
	for (;;)
	{
		cv::Mat frame = cam.GetFrame().clone();
		int progress = calib.GetProgress();
		std::strstream msg;
		std::strstream camStatus;
		if (isMirrored) {
			camStatus << "[MIRRORED] ";
		}
		if (progress == 0) {
			msg << "Camera calibration. Show checkboard to the camera.";
		}
		else if (progress < 100) {
			std::strstream msgstr;
			msg << "Nice! " << progress << "% of calibration complete. Keep going.";
			std::vector<cv::Point2f> c;
			bool found = chk.GetCheckboard(c);
			cv::drawChessboardCorners(frame, cv::Size(6, 9), c, found);
		}
		else {
			CameraIntrinsic intr;
			calib.GetIntrinsic(intr);
			std::strstream msgstr;
			msg << "Calibration complete. Have fun with the cube!";
			if (!undistortionActive)
				camStatus << "[DISTORTED] ";
			if (undistortionActive) {
				cv::Mat undist;
				cv::undistort(frame, undist, intr.cameraMatrix, intr.distCoeffs);
				frame = undist;
			}
			cv::Mat worldMatrix;
			if (calib.CalcExtrinsic(worldMatrix)) {
				cv::Mat tansitionMatrix = intr.cameraMatrix * worldMatrix;
				DrawAxis(frame, tansitionMatrix, 5);
				DrawCube(frame, tansitionMatrix, cv::Point3f(0, 0, 0), 2);
			}
		}
		if (isMirrored) {
			cv::Mat flip;
			cv::flip(frame, flip, 1);
			frame = flip;
		}
		msg << std::ends;
		camStatus << std::ends;
		ShowMessage(frame, msg.str(), cv::Point(10, 20));
		ShowMessage(frame, camStatus.str(), cv::Point(10, 40));
		imshow(windowName, frame);
		int key = cv::waitKey(30);
		if (key == 27) break;
		if (key == 100) undistortionActive = !undistortionActive;
		if (key == 109) isMirrored = !isMirrored;
	}
	if (calib.GetProgress() != 100) calib.TerminateCalibration();
	return 0;
}
