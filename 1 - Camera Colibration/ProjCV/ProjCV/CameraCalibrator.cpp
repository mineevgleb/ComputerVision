#include "CameraCalibrator.h"

//Constructor for CameraCalibrator class
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

//Method for realtime camera calibration
//Captures FRAMES_TO_CAPTURE frames with CAPTURE_DELAY_IN_MS delay between two frames
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
		m_intrinsic.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
		m_intrinsic.distCoeffs = cv::Mat::zeros(8, 1, CV_64F);
		cv::calibrateCamera(m_objectPoints, m_capturedPoints, m_imgSize, m_intrinsic.cameraMatrix,
			m_intrinsic.distCoeffs, cv::noArray(), cv::noArray(), CV_CALIB_RATIONAL_MODEL);
		m_intrinsic.cameraMatrix = 
			cv::getOptimalNewCameraMatrix(m_intrinsic.cameraMatrix, m_intrinsic.distCoeffs, m_imgSize, 0);
		m_progress.store(100);

	});
	
}

//Method that terminates the camera calibration
//Used to prevent failure, when window is closed during the calibration. 
void CameraCalibrator::TerminateCalibration()
{
	m_calibrating = false;
	m_calibThread->join();
	delete m_calibThread;
}

//Method that returns the current progress of the calibration
int CameraCalibrator::GetProgress()
{
	return m_progress.load();
}

//Method that gets the intrinsic values for the camera
bool CameraCalibrator::GetIntrinsic(CameraIntrinsic &out)
{
	if (m_progress.load() == 100) {
		out = m_intrinsic;
		return true;
	}
	else
		return false;
}

//Method that calculates the extrinisc values for the camera
bool CameraCalibrator::CalcExtrinsic(cv::Mat &out)
{
	if (m_progress.load() == 100) {
		std::vector<cv::Point2f> corners;
		bool found = m_checkboard->GetCheckboard(corners);
		if (!found) return false;
		cv::Mat rvec;
		cv::Mat tvec;
		cv::solvePnP(m_objectPoints[0], corners, m_intrinsic.cameraMatrix, m_intrinsic.distCoeffs,
			rvec, tvec);
		cv::Mat rmat;
		cv::Rodrigues(rvec, rmat);
		cv::hconcat(rmat, tvec, out);
		return true;
	}
	else
		return false;
}

//Method that fills a vector with names of all xml files in Calibrations folder.
//Returns true if vector is not empty.
bool CameraCalibrator::ListCalibrationFiles(std::vector<std::string> &out) {
	boost::filesystem::path dir("Calibrations");
	bool result = false;
	if (boost::filesystem::exists(dir))
	{
		if (boost::filesystem::is_directory(dir))
		{
			auto it = boost::filesystem::directory_iterator(dir);
			auto end = boost::filesystem::directory_iterator();
			for (;it != end; ++it) {
				if (it->path().extension() == ".xml") {
					out.push_back(it->path().filename().string());
					result = true;
				}
			}
		}
	}
	return result;
}

//Method that loads intrinsic camera parameters from clibration file
bool CameraCalibrator::CalibrateFromFile(std::string &fileName)
{
	std::stringstream calibrationName;
	calibrationName << "Calibrations\\" << fileName;
	cv::FileStorage fs(calibrationName.str(), cv::FileStorage::READ);
	fs.open(calibrationName.str(), cv::FileStorage::READ);
	if (fs.isOpened()) {
		fs["cameraMatrix"] >> m_intrinsic.cameraMatrix;
		fs["distCoeffs"] >> m_intrinsic.distCoeffs;

		fs.release();
		m_progress.store(100);

		return true;
	}
	else {
		std::cout << "Calibration filename is wrong! Recalibrating...";
		return false;
	}

}

//Method that stores intrinsic parameters for current calibration to a file
void CameraCalibrator::SaveCalibration(std::string &fileName)
{
	boost::filesystem::create_directory("Calibrations");
	std::stringstream calibrationName;
	calibrationName << "Calibrations\\" << fileName << ".xml";
	cv::FileStorage fs(calibrationName.str(), cv::FileStorage::WRITE);
	fs.open(calibrationName.str(), cv::FileStorage::WRITE);
	fs << "cameraMatrix" << m_intrinsic.cameraMatrix;
	fs << "distCoeffs" << m_intrinsic.distCoeffs;
	fs.release();

	std::cout << "Calibration saved: " << fileName << ".xml\n";

}