#include "FaceDetector.h"
#include <boost/filesystem.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <random>
#include "MySVM.h"

namespace nl_uu_science_gmt 
{
	/**
	* Check if metatadta says that there are persons on the image
	*
	* INPUT   path: A path to the metadata file
	*/
	bool hasPerson(std::string path) {
		std::ifstream is(path, std::ifstream::in);
		boost::property_tree::ptree pt;
		boost::property_tree::read_xml(is, pt);
		BOOST_FOREACH(boost::property_tree::ptree::value_type const& node,
			pt.get_child("annotation")) {
			if (node.first == "object" &&
				node.second.get_child("name").get_value<std::string>() == "person") {
				return true;
			}		}		return false;
	}

	void FaceDetector::prepare
		(const MatVec & pos_examples, 
		const MatVec & neg_examples, 
		const double factor, 
		cv::Mat & Xt32F, 
		cv::Mat & Xv32F, 
		cv::Mat & Lt32S, 
		cv::Mat & Lv32S,
		SVMModel &model)
	{
		int maxRows = pos_examples.size() + neg_examples.size();
		int cols = pos_examples[0].cols * pos_examples[0].rows;
		Xt32F = cv::Mat(maxRows, cols, CV_32FC1);
		Xv32F = cv::Mat(maxRows, cols, CV_32FC1);
		Lt32S = cv::Mat(maxRows, 1, CV_32SC1);
		Lv32S = cv::Mat(maxRows, 1, CV_32SC1);
		int tRows = 0;
		int vRows = 0;
		model.neg_offset = 0;
		std::random_device rd;
		std::mt19937 gen(1337);
		std::uniform_real_distribution<> distr(0, 1);
		for (auto img : pos_examples) {
			cv::Mat m = img.reshape(1, 1);
			if (distr(gen) < 1.0 / (factor + 1)) {
				m.row(0).copyTo(Xv32F.row(vRows));
				Lv32S.at<int>(vRows++, 0) = 1;
			}
			else {
				m.row(0).copyTo(Xt32F.row(tRows));
				Lt32S.at<int>(tRows++, 0) = 1;
				++model.neg_offset;
			}
		}
		for (auto img : neg_examples) {
			cv::Mat m = img.reshape(1, 1);
			if (distr(gen) < 1.0 / (factor + 1)) {
				m.row(0).copyTo(Xv32F.row(vRows));
				Lv32S.at<int>(vRows++, 0) = -1;
			}
			else {
				m.row(0).copyTo(Xt32F.row(tRows));
				Lt32S.at<int>(tRows++, 0) = -1;
			}
		}
		Xt32F.resize(tRows);
		Xv32F.resize(vRows);
		Lt32S.resize(tRows);
		Lv32S.resize(vRows);
	}

	FaceDetector::FaceDetector
		(const std::string & path_pos, 
			const std::string & path_neg, 
			const std::string & path_meta,
			const cv::Size & input_size, 
			const int cell_size, 
			const cv::Rect & crop_window, 
			const int max_images,
			const double pos2neg_ratio) : 
		m_model_size(input_size),
		m_cell_size(cell_size),
		m_crop(crop_window)
	{
		int imgReadPos = 0;
		int imgReadNeg = 0;
		for (boost::filesystem::recursive_directory_iterator it(path_pos),
			end; it != end; ++it) {
			std::string name = it->path().string();
			if (it->path().extension() == ".jpg") {
				m_img_fns_pos.push_back(it->path());
				if ((++imgReadPos) >= max_images) break;
			}
		}
		for (boost::filesystem::recursive_directory_iterator it(path_meta),
			end; it != end; ++it) {
			std::string name = it->path().string();
			if (it->path().extension() == ".xml") {
				if (!hasPerson(it->path().string())) {
					m_img_fns_neg.push_back(path_neg + it->path().stem().string() + ".jpg");
					if ((++imgReadNeg) >= imgReadPos * pos2neg_ratio) break;
				}
			}
		}
		int i = 0;
		while ((++imgReadNeg) <= imgReadPos * pos2neg_ratio)
		{
			m_img_fns_neg.push_back(m_img_fns_neg[i++]);
		}	
	}

	void FaceDetector::load(MatVec & images, const bool is_positive)
	{
		srand(42);
		cv::Rect crop = m_crop;
		PathVec &paths = m_img_fns_pos;
		if (!is_positive) {
			paths = m_img_fns_neg;
		}
		for (auto p : paths) {
			cv::Mat img = cv::imread(p.string(), CV_LOAD_IMAGE_GRAYSCALE);
			if (!is_positive) {
				crop = cv::Rect(rand() % (img.cols - 80), rand() % (img.rows - 80), 64, 64);
			}
			cv::Mat result;
			cv::resize(img(crop), result, m_model_size);
			images.push_back(result);
		}
	}

	void FaceDetector::normalize(const MatVec & images, MatVec & features)
	{
		for (auto img : images) {
			cv::Mat normalized;
			img.convertTo(normalized, cv::DataType<float>::type, 1 / 255.0);
			cv::Scalar mean, stddev;
			cv::meanStdDev(normalized, mean, stddev);
			normalized -= mean[0];;
			normalized /= stddev[0];
			features.push_back(normalized);
		}
	}

	void FaceDetector::svmFaces(const MatVec & pos_data, const MatVec & neg_data, SVMModel & model)
	{
		MySVM svm;
		cv::SVMParams params;
		params.svm_type = cv::SVM::C_SVC;
		params.kernel_type = cv::SVM::POLY;
		params.degree = 1;
		params.term_crit = cv::TermCriteria(CV_TERMCRIT_ITER, 100000, 1e-6);
		params.C = 0.6; //!!!!MODIFY!!!!//
		cv::Mat Xt32F, Xv32F, Lt32S, Lv32S;
		prepare(pos_data, neg_data, 80.0 / 20.0, Xt32F, Xv32F, Lt32S, Lv32S, model);
		
		svm.train(Xt32F, Lt32S, cv::Mat(), cv::Mat(), params);
		

		const int sv_count = svm.get_support_vector_count();
		const int sv_length = svm.get_var_count();
		CvSVMDecisionFunc *decision = svm.getDecisionFunc();

		cv::namedWindow("a");
		model.weights.push_back(cv::Mat(sv_length, 1, CV_32FC1, cv::Scalar::all(0)));
		volatile double s = 0;
		for (int i = 0; i < sv_count; ++i) {
			const float *sv = svm.get_support_vector(i);
			
			const double weight = decision->alpha[i];
			s += abs(weight);
			for (int j = 0; j < sv_length; ++j) {
				model.weights[0].at<float>(j, 0) -= sv[j] * weight;
			}
		}

		float min = 1.0;
		float max = -1.0;

		for (int i = 0; i < sv_length; ++i) {
			float a = model.weights[0].at<float>(i, 0);
			if (a < min) min = a;
			if (a > max) max = a;
		}
		cv::imshow("a", (model.weights[0].reshape(1, 20) - min) * (1 / (max - min)));

		model.train_scores = Xt32F * model.weights[0] + decision->rho;
		for (int i = 0; i < model.train_scores.rows; ++i) {
			if (abs(abs(model.train_scores.at<float>(i, 0)) - 1.0f) < 1e-6) {
				model.support_vector_idx.insert(i);
			}
		}
		volatile int a = 0;
		a += 2;
		
		//for (int i = 0; i < model.train_scores.rows; ++i) {
		//	std::cout << model.train_scores.at<float>(i, 0) << "\n";
		//}
	}
}
