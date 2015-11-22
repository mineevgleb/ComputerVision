#pragma once
#include "CameraThread.h"
#include <vector>

//Class for checkboard processing in a separate thread.
//Checkboard processing is quite time consuming. 
//Moving it to the separate thread allows application to run smoothly.
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