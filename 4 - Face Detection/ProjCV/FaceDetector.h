/*
 * FaceDetector.h
 *
 *  Created on: 1 Dec 2015
 *      Author: coert
 */

#ifndef FACEDETECTOR_H_
#define FACEDETECTOR_H_

#include <boost/filesystem/path.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>
#include <string>
#include <set>
#include <vector>

#include "FeatureHOG.h"

namespace nl_uu_science_gmt
{
class MySVM;
} /* namespace nl_uu_science_gmt */

namespace cv
{
class Mat;
} /* namespace cv */

//! A vector of paths
typedef std::vector<boost::filesystem::path> PathVec;

//! A vector of matrices
typedef std::vector<cv::Mat> MatVec;

//! A vector Support Vector machine model representing the a feature space
struct SVMModel
{
	/*
	 * A vector SVM C-values to test on the validation data. Different feature
	 * spaces (gray pixel, HOG, BoVW, etc.) requires its own C value. This is
	 * usually determined via optimisation on some validation data
	 */
	std::vector<double> c_values;

	/*
	 * Model weights per channel. This vector represents the weights of the model
	 * per channel. A gray pixel model only has 1 channel, a HOG model has 31,
	 * each representing a bin of a particular HOG cell
	 */
	std::vector<cv::Mat> weights;
	double bias;

	/*
	 * A filter engine for each model channel. Each channel is convolved by the
	 * FilterEngine with some input
	 */
	std::vector<cv::Ptr<cv::FilterEngine>> engine;

	/*
	 * A column vector containing the model scores of each training example
	 * according to the trained model. pos_examples and neg_examples are
	 * concatenated, pos_examples first, followed by neg_examples
	 */
	cv::Mat train_scores;

	/*
	 * The train_scores contain the scores of both positive and negative
	 * examples. The offset determines the split between pos_examples and
	 * neg_examples
	 */
	int neg_offset;

	/*
	 * A set of indexes for the training examples that are support vectors
	 * (scoring between -1 and 1). The index refers to the order in
	 * train_scores
	 */
	std::set<int> support_vector_idx;
};

struct Layer
{
	//! pyramid layer number
	int l;
	//! layer size scaling factor (applying this factor scales back to the relative model size)
	double factor;
	//! the input feature data for the pyramid layer
	cv::Mat features;
	//! the PDF response data for the pyramid layer
	cv::Mat pdf;
};

//! A vector of input layers
typedef std::vector<Layer> ImagePyramid;

struct Candidate
{
	//! the layer of the image pyramid on which the candidate occurred
	int l;
	//! response score for the candidate
	double score;
	//! region of interest in the feature matrix (in feature space) on the pyramid layer
	cv::Rect ftr_roi;
	//! correctly scaled box of the detection location on the original input image (in image space)
	cv::Rect img_roi;
};
typedef std::vector<Candidate> CandidateVec;

namespace nl_uu_science_gmt
{

/**
 * FaceFetector class for training and testing detection of faces in still images
 */
class FaceDetector
{
	const cv::Size m_model_size;
	//! Feature descriptor cell size (for the pixel model this should be 1!)
	const int m_cell_size;
	//! Desired positive images cropping window
	const cv::Rect m_crop;

	//! Postive image filenames
	PathVec m_img_fns_pos;
	//! Negative image filenames
	PathVec m_img_fns_neg;
	//! Negative meta filenames
	PathVec m_img_fns_train;
	//! Testing image filenames
	PathVec m_img_fns_test;

	/**
	 * Helper method to prepare the input. The input has to be split between Training Data
	 *     and Validation Data with a ratio of about: 80% / 20%. The validation data is used
	 *     to optimize parameters during training iterations and is not to be used as training
	 *     data itself. Use validation only as testing input on trained classifier.
	 *
	 * INPUT   pos_examples: a vector of positive images (CV_8UC3)
	 *         neg_examples: a vector of negative images (CV_8UC3)
	 *         factor: a factor deciding the amount ratio between training and validation data (80% / 20%)
	 * OUTPUT  Xt32F: a matrix containing all training feature descriptions (CV_32FC1/CV_64FC1). One descriptor per row.
	 *         Xv32F: a matrix containing all validation feature descriptions (CV_32FC1/CV_64FC1). One descriptor per row.
	 *         Lt16S: a matrix containing all training labels. One label per row R,
	 *             corresponding to the class of the training example at row R (CV_16S)
	 *         Lv16S: a matrix containing all validation labels. One label per row R,
	 *             corresponding to the class of the validation example at row R (CV_16S)
	 *         model: a model to store negative offset
	 */
	void prepare(
			const MatVec &pos_examples, const MatVec &neg_examples, const double factor,
			cv::Mat &Xt32F, cv::Mat &Xv32F, cv::Mat &Lt32S, cv::Mat &Lv32S, SVMModel &model);

public:
	/**
	 * INPUT   path_pos: a path to the location containing all the positive images
	 *         path_neg: a path to the location containing all the negative images
	 *         path_meta: a path to the location containing all the matadata for the negative images
	 *         input_size: a desired input size
	 *         cell_size: model cell size (1 for pixel model)
	 *         crop_window: a desired cropping window for each positive image
	 *         max_images: a maximum image amount to read
	 *         pos2neg_ratio: a ratio factor for negative images to the read positive images (usually read more neg than pos)
	 */
	FaceDetector(
			const std::string &path_pos, const std::string &path_neg, const std::string &path_meta, const cv::Size &input_size,
			const int cell_size, const cv::Rect &crop_window, const int max_images, const double pos2neg_ratio);
	virtual ~FaceDetector() {};

	/**
	 * Loads the image data from the disk given one or more paths
	 *
	 * INPUT   is_positive : loading positive or negative images
	 * OUTPUT  images: a vector of CV_8UC3 images read from the input paths
	 */
	void load(MatVec &images, const bool is_positive);

	/**
	 * Normalizes the training data by decorrelation of the pixel values in an image, by subtracting its mean
	 * value and dividing the input by its standard deviation value.
	 *
	 * INPUT   images: a vector of CV_8UC3 images (range [0..255])
	 * OUTPUT  features: a vector of normalized (gray) CV_32F/CV64F images (range [0..1]), by subtracting the mean from the images
	 *            and dividing them by their standard deviation.
	 *
	 *         cv::Mat imgFl = (matrix of row representations of all training images. One 1D image vector per row)
	 *         imgFl = imgFl - mean_data;
	 *         imgFl = imgFl / stddev_data;
	 *         output[i] = imgFL.row(i).reshape(channels, image_height);
	 */
	void normalize(
			const MatVec &images, MatVec &features);

	/**
	 * Use a feature descriptor to describe all the images and output a vector of corresponding features
	 *
	 * INPUT  images: a vector of CV_8UC3 images (range [0..255])
	 * OUPUT  features: a vector of CV_32F features in case of HOG it is a 31 channel matrix,
	 *           but the channels (HOG bins) are interleaved in the columns
	 *
	 *        Use cv::split and cv::merge together with reshape(..) to manipulate the channels of the
	 *        feature matrix. Eg:
	 *
	 *        cv::Mat feature_matrix(10, 310, CV_32F);
	 *        cv::Mat feature_channel_matrix = feature_matrix.reshape(31, 10); // feature_channel_matrix.size() == cv::Size(10, 10);
	 *        cv::vector<cv::Mat> feature_channels;
	 *        cv::split(feature_channel_matrix, feature_channels); // feature_channels.size() == 31
	 *        ** do stuff on the separated channels **
	 *        cv::merge(feature_channels, feature_channel_matrix);
	 *        feature_matrix = feature_channel_matrix.reshape(1, 10);
	 */
	void describe(
			const MatVec &images, MatVec &features);

	/**
	 * Create a feature pyramid scaling a query image down by a given factor at each subsequent layer
	 *
	 * INPUT:  an input CV_8UC3 image
	 * OUTPUT: a vector of layers containing the scaled feature space representations of the input image,
	 *             scaled down by a factor at each layer.
	 */
	void createPyramid(
			const int, const cv::Mat &, ImagePyramid &);

	/**
	 * Train a Support Vector Machine given a set of positive and negative training data
	 *
	 * INPUT   pos_data: a row vector of input matrices containing the positive image representations
	 *             each row in pos_data should contain a 1D vector representing 1 input example
	 *         neg_data: a row vector of input matrices containing the negative image representations
	 *             each row in pos_data should contain a 1D vector representing 1 input example
	 * OUTPUT: a model containing the trained filter representations for each channel as a cv::FilterEngine,
	 *             as well as the training scores (see struct SVMModel)
	 */
	void svmFaces(
			const MatVec &pos_data, const MatVec &neg_data, SVMModel &model);

	/**
	 * Discrete Matrix Convolution of a set of filters with a feature descriptor. This should be done by
	 * means of cv::FilterEngine.apply(...), which applies the convolution of a filter to a feature
	 * descriptor matrix.
	 * The result is a Probability Density Function (PDF) (or Response Matrix, RM), which has a score at
	 * every [x, y] location, which represents how well the filter matches the features at that point.
	 *
	 * INPUT:  model: input model, see struct SVMModel
	 *         pyramid_layer: a pyramid layer containing the features of the input to be convolved with
	 *             the Filter Engines in the model
	 * OUTPUT: response_layer: response_layer: a response layer containing the response result as the sum of the
	 *             convolution operations of each channel on the given input layer.
	 */
	void convolve(
			SVMModel &model, Layer &pyramid_layer);

	/**
	 * Converts the Response Matrix to a set of scored Candidate detections, sorted by score (good -> bad)
	 *
	 * INPUT   pyramid: a pyramid as vector of matrices, one for every pyramid layer
	 *         threshold: a threshold above which the candidate must score to be added to candidates
	 * OUTPUT  candidates: a sorted list of candidate detections, containing the response score
	 *             from the scaled Region Of Interest corresponding to the detection position
	 *             such that the ROI has the correct detection size with regard to the pyramid
	 *             layer the detection was found at.
	 */
	void positionalContent(
			const Layer &pyramid_layer, const double threshold, CandidateVec &candidates);

	/**
	 * Non-Maxima-Suppression is a technique to filter out lower scoring candidates that overlap by
	 * a certain amount (following a Union-Over-Intersection measure) with the bounding box of a
	 * higher scoring candidate.
	 *
	 * INPUT         image_size: The size of the queried image
	 * INPUT/OUTPUT  candidates: The vector of candidates filtered on detections that overlap too much with
	 *                    each other, if the overlap (Union Over Intersection) is over a set threshold,
	 *                    only the candidate with the highest score is retained
	 */
	void nonMaximaSuppression(
			const cv::Size &image_size, CandidateVec &candidates);

	/**
	 * Getter for training paths vector
	 *
	 * OUTPUT:  Vector of training paths
	 */
	const PathVec& getTrainFiles() const
	{
		return m_img_fns_train;
	}

	/**
	 * Getter for negative paths vector
	 *
	 * OUTPUT:  Vector of negative image paths
	 */
	const PathVec& getNegImageFiles() const
	{
		return m_img_fns_neg;
	}
};

} /* namespace nl_uu_science_gmt */

#endif /* FACEDETECTOR_H_ */
