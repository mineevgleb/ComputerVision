#include "CameraThread.h"

CameraThread::CameraThread(int deviceNumber, bool run) :
	m_capture(deviceNumber),
	m_isRunning(run),
	m_currentThread(nullptr)
{
	boost::this_thread::sleep(boost::posix_time::seconds(1));
	if (m_isRunning) {Run();}
}

CameraThread::~CameraThread()
{
	Stop();
}

void CameraThread::Run()
{
	m_capture >> m_frame;
	if (!m_currentThread) {
		m_currentThread =
		new boost::thread([this]() {
			while (m_isRunning.load()) {
				UpdateFrame();
			}
		});
	}
}

void CameraThread::Stop()
{
	if (m_currentThread) {
		m_isRunning.store(false);
		m_currentThread->join();
		delete m_currentThread;
		m_currentThread = nullptr;
	}
}

cv::Mat CameraThread::GetFrame()
{
	boost::shared_lock<boost::shared_mutex> lock(m_mutex);
	return m_frame;
}

void CameraThread::UpdateFrame() {
	cv::Mat frame;
	m_capture >> frame;
	boost::unique_lock<boost::shared_mutex> lock(m_mutex);
	m_frame = frame;
}