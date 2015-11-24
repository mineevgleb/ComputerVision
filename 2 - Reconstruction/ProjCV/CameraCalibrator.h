#pragma once
#include <opencv2\opencv.hpp>
#include <string>
#include <strstream>
#include <vector>
#include <ctime>

struct CameraIntrinsic {
	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
};

//Class, used for real-time camera calibration, as well as for saving and loading intinsic parameters.
class CameraCalibrator {
public:
	CameraCalibrator(const char* calibVideo, cv::Size &imgSize, cv::Size cbSize);
	void Calibrate();
	bool GetIntrinsic(CameraIntrinsic &out);
	void SaveCalibration(std::string &fileName);
	static const int FRAMES_TO_SKIP = 100;
private:
	cv::Size m_imgSize;
	cv::Size m_cbSize;
	CameraIntrinsic m_intrinsic;
	cv::VideoCapture m_capt;
	int m_capturedAmount;
	std::vector<std::vector<cv::Point3f>> m_objectPoints;
	std::vector<std::vector<cv::Point2f>> m_capturedPoints;
	bool m_calibrating;
};