#include "CameraCalibrator.h"

//Constructor for CameraCalibrator class
CameraCalibrator::CameraCalibrator(const char* calibVideo, cv::Size &imgSize, cv::Size cbSize) :
	m_capt(),
	m_imgSize(imgSize),
	m_cbSize(cbSize)
{
	m_capt.open(calibVideo);
	m_objectPoints.push_back(std::vector<cv::Point3f>());
	for (int i = 0; i < m_cbSize.height; ++i) {
		for (int j = 0; j < m_cbSize.width; ++j) {
			m_objectPoints[0].push_back(cv::Point3f(j, i, 0));
		}
	}
}

//Method for camera calibration
void CameraCalibrator::Calibrate()
{
	m_capturedAmount = 0;
	m_capturedPoints.clear();
	m_calibrating = true;
	m_capturedPoints.push_back(std::vector<cv::Point2f>());
	cv::Mat cur;
	while (m_capt.read(cur)) {
		bool found = cv::findChessboardCorners(cur,
			m_cbSize, m_capturedPoints[m_capturedAmount],
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE | CV_CALIB_CB_FILTER_QUADS);
		if (found) {
			cv::Mat frameGray;
			cv::cvtColor(cur, frameGray, cv::COLOR_BGR2GRAY);
			cv::cornerSubPix(frameGray, m_capturedPoints[m_capturedAmount], cv::Size(11, 11),
				cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			//cv::drawChessboardCorners(cur, m_cbSize, m_capturedPoints[m_capturedAmount], found);
			//cv::namedWindow("found");
			//cv::imshow("found", cur);
			//int a = cv::waitKey();
			//if (a == 'y') {
				m_capturedAmount++;
				m_capturedPoints.push_back(std::vector<cv::Point2f>());
			//}
			
			for (int i = 0; i < FRAMES_TO_SKIP; ++i) m_capt.read(cur);
		}	
		m_capt.read(cur);
		m_capt.read(cur);
		m_capt.read(cur);
	}
	m_intrinsic.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
	m_intrinsic.distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
	m_objectPoints.resize(m_capturedAmount, m_objectPoints[0]);
	m_capturedPoints.pop_back();
	cv::calibrateCamera(m_objectPoints, m_capturedPoints, m_imgSize, m_intrinsic.cameraMatrix,
		m_intrinsic.distCoeffs, cv::noArray(), cv::noArray());
	//m_intrinsic.cameraMatrix = 
		//cv::getOptimalNewCameraMatrix(m_intrinsic.cameraMatrix, m_intrinsic.distCoeffs, m_imgSize, 0);
}

//Method that gets the intrinsic values for the camera
bool CameraCalibrator::GetIntrinsic(CameraIntrinsic &out)
{
	out = m_intrinsic;
	return true;
}

//Method that stores intrinsic parameters for current calibration to a file
void CameraCalibrator::SaveCalibration(std::string &fileName)
{
	cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
	fs.open(fileName, cv::FileStorage::WRITE);
	cv::Mat m = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat d = cv::Mat::zeros(3, 3, CV_32F);
	m_intrinsic.cameraMatrix.convertTo(m, CV_32F);
	m_intrinsic.distCoeffs.convertTo(d, CV_32F);
	fs << "CameraMatrix" << m;
	fs << "DistortionCoeffs" << d;
	fs.release();

	std::cout << "Calibration saved: " << fileName << std::endl;

}