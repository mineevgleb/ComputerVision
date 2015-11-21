#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include <strstream>
#include <vector>
#include <ctime>
#include <boost\filesystem.hpp>
#include "CameraThread.h"
#include "CheckboardThread.h"
#include "CameraCalibrator.h"
#include "Draw.h"

#define DRAW_COMPLEX

//Method that shows a text on the screen starting from a certain position point
void ShowMessage(cv::Mat &img, const char *msg, cv::Point pos) {
	cv::putText(img, msg, cv::Point(pos.x + 1, pos.y + 1), CV_FONT_HERSHEY_SIMPLEX,
		0.5, cv::Scalar(0, 0, 0, 255), 1);
	cv::putText(img, msg, pos, CV_FONT_HERSHEY_SIMPLEX,
		0.5, cv::Scalar(50, 50, 200, 255), 1, CV_AA);
}

//Method that calls the save calibration method from CameraCalibrator
void SaveCalibration(CameraCalibrator &calib) {
	time_t seconds;
	time(&seconds);
	tm *t = localtime(&seconds);
	t->tm_year += 1900;
	std::stringstream calibname;
	calibname << t->tm_mday << "." << t->tm_mon << "." << t->tm_year << " "
		<< t->tm_hour << "-" << t->tm_min << "-" << t->tm_sec;
	calib.SaveCalibration(calibname.str());
}

//Main method; contains logic for asking for input and a continuous loop that show the opencv output 
int main(int argc, char **argv) {
	CameraThread cam(0, true);
	CheckboardThread chk(&cam, true);
	CameraCalibrator calib(&chk, cam.GetFrame().size());
	std::string filename;

	char answer = 0;
	std::vector<std::string> calibFiles;
	if (calib.ListCalibrationFiles(calibFiles)) {
		std::cout << "Calibration presets found. Would you like to use one of them? ([y]/[n])\n";
		while (true) {
			std::cin >> answer;
			if (answer != 'n' && answer != 'N' && answer != 'y' && answer != 'Y') {
				std::cout << "Invalid input! Press [y] to use preset or [n] to recalibrate.\n";
			}
			else {
				break;
			}
		}
	}
	else {
		answer = 'n';
	}

	if (answer == 'n') {
		calib.Calibrate();
	}
	else {
		std::cout << "Enter the number of preset file you want to use:\n";
		for (int i = 0; i < calibFiles.size(); ++i) {
			std::cout << i + 1 << ":\t" << calibFiles[i] << "\n";
		}
		int num = 0;
		while (true) {
			std::cin >> num;
			if (num < 1 || num > calibFiles.size()) {
				std::cout << "Invalid input. Enter the number of preset file you want to use.\n";
			}
			else {
				break;
			}
		}
		if (!calib.CalibrateFromFile(calibFiles[num - 1])) {
			calib.Calibrate();
		}
	}
	const char *windowName = "Calibration";
	cv::namedWindow(windowName);
	bool undistortionActive = true;
	bool isMirrored = true;
	bool takeScreenshot = false;
	for (;;)
	{
		cv::Mat frame = cam.GetFrame().clone();
		int progress = calib.GetProgress();
		std::strstream msg;
		std::strstream useInfo;
		std::strstream useInfo2;
		std::strstream camStatus;
		if (isMirrored) {
			camStatus << "[MIRRORED] ";
		}
		useInfo << "[esc] - exit, [m] - mirror, [s] - screenshot";
		if (progress == 0) {
			msg << "Camera calibration. Show checkboard to the camera.";
		}
		else if (progress < 100) {
			if (progress == (CameraCalibrator::FRAMES_TO_CAPTURE * 100) 
				/ (CameraCalibrator::FRAMES_TO_CAPTURE + 1)) {
				msg << "Calibrating... Wait a bit.";
			}
			else {
				msg << "Nice! " << progress << "% of frames captured. Keep moving the picture for proper calibration.";
			}
			std::vector<cv::Point2f> c;
			bool found = chk.GetCheckboard(c);
			cv::drawChessboardCorners(frame, cv::Size(6, 9), c, found);
		}
		else {
			if (answer == 'n') {
				SaveCalibration(calib);
				answer = 'y';
			}
			CameraIntrinsic intr;
			calib.GetIntrinsic(intr);
			msg << "Calibration complete. Have fun with the cube!";
			useInfo2 << "[d] - distortion";
			if (!undistortionActive)
				camStatus << "[DISTORTED] ";
			if (undistortionActive) {
				cv::Mat undist;
				cv::undistort(frame, undist, intr.cameraMatrix, intr.distCoeffs);
				frame = undist;
			}
			cv::Mat worldMatrix;
			if (calib.CalcExtrinsic(worldMatrix)) {
#ifdef DRAW_COMPLEX
				std::vector<LineWithDepth> lines;
				GetCubeLines(intr.cameraMatrix, worldMatrix, cv::Point3f(0, 0, 0), 2,
					cv::Scalar(13, 213, 252, 255), 5, lines);
				GetAxisLines(intr.cameraMatrix, worldMatrix, 2, 5, 3, lines);
				std::vector<CircleWithDepth> circles;
				decomposeLines(lines, circles, frame);
				drawCircles(circles, frame);

#else
				cv::Mat tansitionMatrix = intr.cameraMatrix * worldMatrix;
				DrawAxis(frame, tansitionMatrix, 5);
				DrawCube(frame, tansitionMatrix, cv::Point3f(0, 0, 0), 2);
#endif
			}
		}
		if (isMirrored) {
			cv::Mat flip;
			cv::flip(frame, flip, 1);
			frame = flip;
		}
		if (takeScreenshot) {
			time_t seconds;
			time(&seconds);
			tm *t = localtime(&seconds);
			t->tm_year += 1900;
			std::cout << "Screen captured: " <<
				t->tm_mday << "." << t->tm_mon << "." << t->tm_year << " "
				<< t->tm_hour << ":" << t->tm_min << ":" << t->tm_sec << std::endl;
			std::stringstream screenname;
			boost::filesystem::create_directory("Screens");
			screenname << "Screens\\" << t->tm_mday << "." << t->tm_mon << "." << t->tm_year << " "
				<< t->tm_hour << "-" << t->tm_min << "-" << t->tm_sec << ".jpg";
			cv::imwrite(screenname.str(), frame);
			takeScreenshot = false;
		}
		msg << std::ends;
		useInfo << std::ends;
		useInfo2 << std::ends;
		camStatus << std::ends;
		ShowMessage(frame, msg.str(), cv::Point(10, 20));
		ShowMessage(frame, camStatus.str(), cv::Point(10, 40));
		ShowMessage(frame, useInfo.str(), cv::Point(10, frame.size().height - 30));
		ShowMessage(frame, useInfo2.str(), cv::Point(10, frame.size().height - 10));
		imshow(windowName, frame);
		int key = cv::waitKey(30);
		if (key == 27) break;
		if (key == 100) undistortionActive = !undistortionActive;
		if (key == 109) isMirrored = !isMirrored;
		if (key == 115) takeScreenshot = true;
	}
	if (calib.GetProgress() != 100) calib.TerminateCalibration();
	return 0;
}
