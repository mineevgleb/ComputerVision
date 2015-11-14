#pragma once
#include "CameraThread.h"
#include <vector>

//Class for thread safe image capturing from camera
class CheckboardThread {
public:
	CheckboardThread(CameraThread *cam, bool run);
	~CheckboardThread();
	void Run();
	void Stop();
	bool GetCheckboard(std::vector<cv::Point2f> &out);
	static const int POINTS_PER_ROW = 6;
	static const int POINTS_PER_COL = 9;
private:
	void UpdateCheckboard();
	std::vector<cv::Point2f> m_checkboard;
	bool m_found;
	CameraThread *m_cam;
	boost::thread *m_currentThread;
	std::atomic_bool m_isRunning;
	boost::shared_mutex m_mutex;
};