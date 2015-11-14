#include "CheckboardThread.h"

#include "CameraThread.h"

CheckboardThread::CheckboardThread(CameraThread *cam, bool run) :
	m_cam(cam),
	m_isRunning(run),
	m_currentThread(nullptr)
{
	if (m_isRunning) { Run(); }
}

CheckboardThread::~CheckboardThread()
{
	Stop();
}

void CheckboardThread::Run()
{
	UpdateCheckboard();
	if (!m_currentThread) {
		m_currentThread =
			new boost::thread([this]() {
			while (m_isRunning.load()) {
				UpdateCheckboard();
				boost::this_thread::sleep(boost::posix_time::milliseconds(1000 / 60));
			}
		});
	}
}

void CheckboardThread::Stop()
{
	if (m_currentThread) {
		m_isRunning.store(false);
		m_currentThread->join();
		delete m_currentThread;
		m_currentThread = nullptr;
	}
}

void CheckboardThread::UpdateCheckboard()
{
	cv::Mat frame = m_cam->GetFrame();
	std::vector<cv::Point2f> tmp;
	bool found = cv::findChessboardCorners(frame,
		cv::Size(POINTS_PER_ROW, POINTS_PER_COL), tmp,
		CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE | CV_CALIB_CB_FILTER_QUADS);
	if (found) {
		cv::Mat frameGray;
		cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
		cv::cornerSubPix(frameGray, tmp, cv::Size(11, 11),
			cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
	}
	boost::unique_lock<boost::shared_mutex> lock(m_mutex);
	m_found = found;
	m_checkboard = tmp;
}

bool CheckboardThread::GetCheckboard(std::vector<cv::Point2f> &out)
{
	boost::shared_lock<boost::shared_mutex> lock(m_mutex);
	out = m_checkboard;
	return m_found;
}