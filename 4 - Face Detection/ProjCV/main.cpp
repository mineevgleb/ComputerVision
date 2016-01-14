#include "FaceDetector.h"
#include <opencv2\highgui\highgui.hpp>

using namespace nl_uu_science_gmt;

int main() {
	FaceDetector detector(".\\data\\positive\\",
		".\\data\\negative\\JPEGImages\\",
		".\\data\\negative\\Annotations\\",
		cv::Size(20, 20), 1,
		cv::Rect(85, 95, 80, 80),
		100, 2.5);
	MatVec pos;
	MatVec neg;
	detector.load(pos, true);
	detector.load(neg, false);
	MatVec pos_norm;
	MatVec neg_norm;
	detector.normalize(pos, pos_norm);
	detector.normalize(neg, neg_norm);
	SVMModel model;
	detector.svmFaces(pos_norm, neg_norm, model);
	cv::waitKey();
	return 0;
}