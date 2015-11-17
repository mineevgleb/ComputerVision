#include "Draw.h"

void GetCubeLines(const cv::Mat &camMat, const cv::Mat &worldMat,
	const cv::Point3f &origin, float edgeSize, cv::Scalar &col, float width, std::vector<LineWithDepth> &out) {
	cv::Point3f cube[8];
	for (int i = 0; i < 8; i++)
		cube[i] = origin;
	cube[1].x += edgeSize;
	cube[2].x += edgeSize; cube[2].y += edgeSize;
	cube[3].y += edgeSize;
	cube[4].z -= edgeSize;
	cube[5].x += edgeSize; cube[5].z -= edgeSize;
	cube[6].x += edgeSize; cube[6].y += edgeSize; cube[6].z -= edgeSize;
	cube[7].y += edgeSize; cube[7].z -= edgeSize;
	PointWithDepth cubeOnScreen[8];
	for (int i = 0; i < 8; ++i)
		cubeOnScreen[i] = PointWithDepth(cube[i], worldMat, camMat);
	for (int i = 0; i < 4; ++i) {
		out.push_back(LineWithDepth(cubeOnScreen[i], cubeOnScreen[(i + 1) % 4], col, width));
	}
	for (int i = 0; i < 4; ++i) {
		out.push_back(LineWithDepth(cubeOnScreen[i + 4], cubeOnScreen[(i + 1) % 4 + 4], col, width));
	}
	for (int i = 0; i < 4; ++i) {
		out.push_back(LineWithDepth(cubeOnScreen[i], cubeOnScreen[i + 4], col, width));
	}
}

void GetAxisLines(const cv::Mat &camMat, const cv::Mat &worldMat,
	float startFrom, float length, float width, std::vector<LineWithDepth> &out) {
		cv::Point3f axis[6];
		for (int i = 0; i < 6; i++)
			axis[i] = cv::Point3f(0, 0, 0);
		axis[0].x = startFrom;
		axis[1].x = length;
		axis[2].y = startFrom;
		axis[3].y = length;
		axis[4].z = -startFrom;
		axis[5].z = -length;
		PointWithDepth axisOnScreen[6];
		for (int i = 0; i < 6; ++i)
			axisOnScreen[i] = PointWithDepth(axis[i], worldMat, camMat);
		out.push_back(LineWithDepth(axisOnScreen[0], axisOnScreen[1], cv::Scalar(0, 0, 255, 255), 2));
		out.push_back(LineWithDepth(axisOnScreen[2], axisOnScreen[3], cv::Scalar(0, 255, 0, 255), 2));
		out.push_back(LineWithDepth(axisOnScreen[4], axisOnScreen[5], cv::Scalar(255, 0, 0, 255), 2));
}

void decomposeLines(std::vector<LineWithDepth> &in, std::vector<CircleWithDepth> &out, cv::Mat &img) {
	for (LineWithDepth &l : in) {
		cv::LineIterator it(img, l.p1.pt, l.p2.pt);
		for (int i = 0; i < it.count; ++i, ++it) {
			float progress = float(i) / it.count;
			float curDepth = l.p1.d * (1 - progress) + l.p2.d * progress;
			out.push_back(CircleWithDepth(PointWithDepth(it.pos(), curDepth), l.col, l.width / 2));
		}
	}
}

void drawCircles(std::vector<CircleWithDepth> &in, cv::Mat &img) {
	std::sort(in.begin(), in.end());
	float maxd = in.begin()->p.d;
	float mind = in.rbegin()->p.d;
	for (CircleWithDepth &c : in) {
		cv::Scalar currentCol = cv::Scalar(c.col * ((2 - (c.p.d - mind) / (maxd - mind))) / 2);
		cv::circle(img, c.p.pt, c.rad, currentCol, -1);
	}
}

cv::Point2f TranslatePointToScreen(const cv::Point3f &pt, const cv::Mat &translationMatrix) {
	cv::Mat ptMat = (cv::Mat_<double>(4, 1) << pt.x, pt.y, pt.z, 1);
	cv::Mat onscreenMat = translationMatrix * ptMat;
	return cv::Point2f(onscreenMat.at<double>(0) / onscreenMat.at<double>(2),
		onscreenMat.at<double>(1) / onscreenMat.at<double>(2));
}

void DrawSimpleCube(cv::Mat &frame, const cv::Mat & translationMatrix, const cv::Point3f &origin, float edgeSize)
{
	cv::Point3f cube[8];
	for (int i = 0; i < 8; i++)
		cube[i] = origin;
	cube[1].x += edgeSize;
	cube[2].x += edgeSize; cube[2].y += edgeSize;
	cube[3].y += edgeSize;
	cube[4].z -= edgeSize;
	cube[5].x += edgeSize; cube[5].z -= edgeSize;
	cube[6].x += edgeSize; cube[6].y += edgeSize; cube[6].z -= edgeSize;
	cube[7].y += edgeSize; cube[7].z -= edgeSize;
	cv::Point2f cubeOnScreen[8];
	for (int i = 0; i < 8; ++i)
		cubeOnScreen[i] = TranslatePointToScreen(cube[i], translationMatrix);
	for (int i = 0; i < 4; ++i) {
		cv::line(frame, cubeOnScreen[i], cubeOnScreen[(i + 1) % 4], cv::Scalar(13, 213, 252, 255), 5);
	}
	for (int i = 0; i < 4; ++i) {
		cv::line(frame, cubeOnScreen[i + 4], cubeOnScreen[(i + 1) % 4 + 4], cv::Scalar(13, 213, 252, 255), 5);
	}
	for (int i = 0; i < 4; ++i) {
		cv::line(frame, cubeOnScreen[i], cubeOnScreen[i + 4], cv::Scalar(13, 213, 252, 255), 5);
	}
}

void DrawSimpleAxis(cv::Mat &frame, const cv::Mat & translationMatrix, float len)
{
	cv::Point3f axis[4];
	for (int i = 0; i < 4; i++)
		axis[i] = cv::Point3f(0, 0, 0);
	axis[1].x = len;
	axis[2].y = len;
	axis[3].z = -len;
	cv::Point2f axisOnScreen[4];
	for (int i = 0; i < 4; ++i)
		axisOnScreen[i] = TranslatePointToScreen(axis[i], translationMatrix);
	cv::line(frame, axisOnScreen[0], axisOnScreen[1], cv::Scalar(0, 0, 255, 255), 2);
	cv::line(frame, axisOnScreen[0], axisOnScreen[2], cv::Scalar(0, 255, 0, 255), 2);
	cv::line(frame, axisOnScreen[0], axisOnScreen[3], cv::Scalar(255, 0, 0, 255), 2);
}