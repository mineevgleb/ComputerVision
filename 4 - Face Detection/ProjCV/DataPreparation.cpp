#include "DataPreparation.h"

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <opencv2\opencv.hpp>
#include <string>
#include <random>


void separateTrainValidSetsPositive(const cv::Size &size, const cv::Rect &crop_window) {
	boost::filesystem::path dir_from(".\\data\\trainvalid_raw\\");
	boost::filesystem::path dir_to_valid(".\\data\\trainvalid\\valid\\positive\\");
	boost::filesystem::path dir_to_train(".\\data\\trainvalid\\train\\positive\\");
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> distr(0, 1);
	for (boost::filesystem::recursive_directory_iterator it(dir_from),
		end; it != end; ++it) {
		std::string name = it->path().string();
		if (it->path().extension() == ".jpg") {
			cv::Mat source = cv::imread(name);
			cv::Mat result;
			cv::resize(source(crop_window), result, size);
			if (distr(gen) < 0.2) {
				cv::imwrite(dir_to_valid.string() + it->path().filename().string(), result);
			}
			else {
				cv::imwrite(dir_to_train.string() + it->path().filename().string(), result);
			}
		}
	}
}



void extractTrainValidSamplesNegative(const cv::Size &size) {
	boost::filesystem::path dir_annot(".\\data\\negative\\Annotations\\");
	boost::filesystem::path dir_img(".\\data\\negative\\JPEGImages\\");
	boost::filesystem::path dir_to_valid(".\\data\\trainvalid\\valid\\negative\\");
	boost::filesystem::path dir_to_train(".\\data\\trainvalid\\train\\negative\\");
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> distr(0, 1);
	int remaining = 12000 * 3;
	while (remaining > 0) {
		for (boost::filesystem::recursive_directory_iterator it(dir_annot),
			end; it != end; ++it) {
			if (hasPerson(it->path().string()))
				continue;

			std::string img_path = dir_img.string() + it->path().stem().string() + ".jpg";

			cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
			cv::Mat result;
			cv::Rect rect(distr(gen) * (img.cols - 80), distr(gen) * (img.rows - 80), 64, 64);
			cv::resize(img(rect), result, size);

			if (distr(gen) < 0.2) {
				cv::imwrite(dir_to_valid.string() +
					it->path().stem().string() + "_" + std::to_string(remaining) + ".jpg", result);
			}
			else {
				cv::imwrite(dir_to_train.string() +
					it->path().stem().string() + "_" + std::to_string(remaining) + ".jpg", result);
			}
			if (!(remaining--)) break;
		}
	}
}