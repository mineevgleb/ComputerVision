#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

const int POINTS_PER_ROW = 6;
const int POINTS_PER_COL = 9;

int main(int argc, char **argv) {
	cv::VideoCapture capture(0);
	if (argc > 1) {
		capture.release();
		capture.open(atoi(argv[1]));
	}
	if (!capture.isOpened()) {
		std::cout << "unable to open video capturing device\n";
		return -1;
	}
	cv::Mat edges;
	cv::namedWindow("Calibration", 1);
	std::vector<cv::Point2f> corners;
	for (;;)
	{
		cv::Mat frame;
		capture >> frame;
		bool found = cv::findChessboardCorners(frame,
			cv::Size(POINTS_PER_ROW, POINTS_PER_COL), corners);
		if (found) {
			cv::cvtColor(frame, frame, CV_RGB2GRAY);
		}
		imshow("Calibration", frame);
		if (cv::waitKey(30) >= 0) break;
	}
	return 0;
}