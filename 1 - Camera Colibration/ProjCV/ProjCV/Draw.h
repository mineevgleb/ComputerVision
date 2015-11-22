#pragma once
#include <opencv2\opencv.hpp>

//Methods, responsible for drawing cube and axis are located in Draw.h and Draw.cpp.

//2D point with attached depth info.
struct PointWithDepth {
	//Constructor, that converts 3D point on board into 2D point on screen, with depth info attached
	PointWithDepth(const cv::Point3f &localPt, const cv::Mat &worldMat, const cv::Mat &camMat) {
		cv::Mat ptMat = (cv::Mat_<double>(4, 1) << localPt.x, localPt.y, localPt.z, 1);
		cv::Mat inWorld = worldMat * ptMat;
		d = inWorld.at<double>(2);
		cv::Mat onScreen = camMat * inWorld;
		pt = cv::Point2f(onScreen.at<double>(0) / onScreen.at<double>(2),
			onScreen.at<double>(1) / onScreen.at<double>(2));
	}
	PointWithDepth(const cv::Point2f &pt, float d) : pt(pt), d(d) {}
	PointWithDepth() : pt(), d() {}
	cv::Point2f pt;
	float d;
};

//2D line with attached depth info for both ends.
struct LineWithDepth {
	PointWithDepth p1;
	PointWithDepth p2;
	cv::Scalar col;
	float width;
	LineWithDepth(const PointWithDepth &p1, const PointWithDepth &p2, const cv::Scalar &col, float width) :
		p1(p1), p2(p2), col(col), width(width) {}
};

//2D circle with attached depth info.
struct CircleWithDepth {
	PointWithDepth p;
	cv::Scalar col;
	float rad;
	CircleWithDepth(const PointWithDepth &p, const cv::Scalar &col, float rad) :
		p(p), col(col), rad(rad) {}
	bool operator < (CircleWithDepth &other) {
		return p.d > other.p.d;
	}
};

void GetCubeLines(const cv::Mat &camMat, const cv::Mat &worldMat,
	const cv::Point3f &origin, float edgeSize, cv::Scalar &col, float width, std::vector<LineWithDepth> &out);

void GetAxisLines(const cv::Mat &camMat, const cv::Mat &worldMat,
	float startFrom, float length, float width, std::vector<LineWithDepth> &out);

void decomposeLines(std::vector<LineWithDepth> &in, std::vector<CircleWithDepth> &out, cv::Mat &img);

void drawCircles(std::vector<CircleWithDepth> &in, cv::Mat &img);

cv::Point2f TranslatePointToScreen(const cv::Point3f &pt, const cv::Mat &translationMatrix);

void DrawSimpleCube(cv::Mat &frame, const cv::Mat & translationMatrix, const cv::Point3f &origin, float edgeSize);

void DrawSimpleAxis(cv::Mat &frame, const cv::Mat & translationMatrix, float len);