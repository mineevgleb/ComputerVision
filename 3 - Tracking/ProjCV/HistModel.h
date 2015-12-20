#pragma once
#include <cstring>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <fstream>
typedef unsigned char uchar;

class HistModel {
public:
	HistModel() : ptAm(0) { memset(hist, 0, sizeof(int) * 8 * 8 * 8); }

	void AddPt(uchar r, uchar g, uchar b) {
		hist[r/32][g/32][b/32]++;
		ptAm++;
	}

	float CalcDiff(HistModel *another) {
		float sum = 0;
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				for (int k = 0; k < 8; k++) {
					sum += abs(((float)hist[i][j][k] / ptAm) - ((float)another->hist[i][j][k] / another->ptAm));
				}
			}
		}
		return sum;
	}

	void Show() {
		cv::Mat graph(500, 512, CV_8UC3, cv::Scalar::all(255));
		int am = 0;
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				for (int k = 0; k < 8; k++) {
					cv::line(graph, cv::Point(am, 500 - (float)hist[i][j][k] / ptAm * 2000),
						cv::Point(am, 500), cv::Scalar(i * 32, j * 32, k * 32));
						am++;
				}
			}
		}
		cv::namedWindow("histogram");
		cv::imshow("histogram", graph);
	}

	void Save(const char * file) {
		std::ofstream of(file);
		of << ptAm << ' ';
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				for (int k = 0; k < 8; k++) {
					of << hist[i][j][k] << ' ';
				}
			}
		}
		of << "/n";
		of.flush();
		of.close();
	}

	void Load(const char * file) {
		std::ifstream ifs(file);
		ifs >> ptAm;
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				for (int k = 0; k < 8; k++) {
					ifs >> hist[i][j][k];
				}
			}
		}
		ifs.close();
	}

protected:
	int hist[8][8][8];
	int ptAm;
};