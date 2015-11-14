#pragma once
#include <opencv2\opencv.hpp>
#include <boost\thread.hpp>
#include <atomic>

//Class for thread safe image capturing from camera
class CameraThread {
public:
	CameraThread(int deviceNumber, bool run);
	~CameraThread();
	void Run();
	void Stop();
	cv::Mat GetFrame();
private:
	void UpdateFrame();
	cv::Mat m_frame;
	cv::VideoCapture m_capture;
	boost::thread *m_currentThread;
	std::atomic_bool m_isRunning;
	boost::shared_mutex m_mutex;
};