#include "CameraCalibrator.h"

CameraCalibrator::CameraCalibrator(CheckboardThread *checkboard, cv::Size &imgSize) :
	m_progress(0),
	m_checkboard(checkboard),
	m_imgSize(imgSize)
{
	m_objectPoints.push_back(std::vector<cv::Point3f>());
	for (int i = 0; i < CheckboardThread::POINTS_PER_COL; ++i) {
		for (int j = 0; j < CheckboardThread::POINTS_PER_ROW; ++j) {
			m_objectPoints[0].push_back(cv::Point3f(j, i, 0));
		}
	}
	m_objectPoints.resize(FRAMES_TO_CAPTURE, m_objectPoints[0]);
}

void CameraCalibrator::Calibrate()
{
	m_progress = 0;
	m_capturedAmount = 0;
	m_capturedPoints.clear();
	m_capturedPoints.resize(FRAMES_TO_CAPTURE);
	m_calibrating = true;
	m_calibThread = new boost::thread([this](){
		while (m_capturedAmount < FRAMES_TO_CAPTURE) {
			if (!m_calibrating) return;
			bool found = m_checkboard->GetCheckboard(m_capturedPoints[m_capturedAmount]);
			if (found) {
				m_capturedAmount++;
				m_progress.store((m_capturedAmount * 100) / (FRAMES_TO_CAPTURE + 1));
				boost::this_thread::sleep(boost::posix_time::milliseconds(CAPTURE_DELAY_IN_MS));
			}	
		}
		std::vector<cv::Mat> rvecs;
		std::vector<cv::Mat> tvecs;
		cv::calibrateCamera(m_objectPoints, m_capturedPoints, m_imgSize, m_intrinsic.cameraMatrix,
			m_intrinsic.distCoeffs, rvecs, tvecs);
		m_progress.store(100);
	});
	
}

void CameraCalibrator::TerminateCalibration()
{
	m_calibrating = false;
	m_calibThread->join();
	delete m_calibThread;
}

int CameraCalibrator::GetProgress()
{
	return m_progress.load();
}

bool CameraCalibrator::GetIntrinsic(CameraIntrinsic &out)
{
	if (m_progress.load() == 100) {
		out = m_intrinsic;
		return true;
	}
	else
		return false;
}

bool CameraCalibrator::CalcExtrinsic(CameraExtrinsic &out)
{
	if (m_progress.load() == 100) {
		std::vector<cv::Point2f> corners;
		bool found = m_checkboard->GetCheckboard(corners);
		if (!found) return false;
		cv::solvePnP(m_objectPoints[0], corners, m_intrinsic.cameraMatrix, m_intrinsic.distCoeffs,
			out.rvec, out.tvec);
		return true;
	}
	else
		return false;
}
