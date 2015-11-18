#pragma once
#include <opencv2\opencv.hpp>
#include <boost\filesystem.hpp>
#include <string>
#include <strstream>
#include <vector>
#include <ctime>
#include "CheckboardThread.h"

struct CameraIntrinsic {
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
};

struct CameraExtrinsic {
	cv::Mat rvec;
	cv::Mat tvec;
};

class CameraCalibrator {
public:
	CameraCalibrator(CheckboardThread *checkboard, cv::Size &imgSize);
	void Calibrate();
	bool CalibrateFromFile(std::string &fileName);
	void SaveCalibration(std::string &fileName);
	void TerminateCalibration();
	int GetProgress();
	bool GetIntrinsic(CameraIntrinsic &out);
	bool CalcExtrinsic(cv::Mat &out);

	static const int FRAMES_TO_CAPTURE = 36;
	static const int CAPTURE_DELAY_IN_MS = 500;
private:
	cv::Size m_imgSize;
	std::atomic<int> m_progress;
	CameraIntrinsic m_intrinsic;
	CameraExtrinsic m_extrinsic;
	CheckboardThread *m_checkboard;
	int m_capturedAmount;
	std::vector<std::vector<cv::Point3f>> m_objectPoints;
	std::vector<std::vector<cv::Point2f>> m_capturedPoints;
	bool m_calibrating;
	boost::thread *m_calibThread;
};