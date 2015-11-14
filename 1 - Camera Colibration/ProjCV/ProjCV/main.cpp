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

int main(int argc, char **argv) {
	CameraThread cam(0, true);
	CheckboardThread chk(&cam, true);
	CameraCalibrator calib(&chk, cam.GetFrame().size());
	calib.Calibrate();
	const char *windowName = "Calibration";
	cv::namedWindow(windowName);
	for (;;)
	{
		cv::Mat frame = cam.GetFrame().clone();
		int progress = calib.GetProgress();
		char *msg = nullptr;
		if (progress == 0) {
			msg = "Camera calibration. Show checkboard to the camera.";
		}
		else if (progress < 100) {
			std::strstream msgstr;
			msgstr << "Nice! " << progress << "% of calibration done. Keep going." << std::ends;
			msg = msgstr.str();
		}
		else {
			msg = "Calibration complete ";
		}
		ShowMessage(frame, msg, cv::Point(10, 20));
		imshow(windowName, frame);
		if (cv::waitKey(30) >= 0) break;
	}
	if (calib.GetProgress() != 100) calib.TerminateCalibration();
	return 0;
}
