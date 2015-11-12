#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <string>

int main() {
	std::cout << "Enter path to the image: ";
	std::string path;
	std::cin >> path;
	cv::Mat img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
	if (!img.data)                              
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	int w = img.cols;
	int h = img.rows;
	cv::rectangle(img, cv::Rect(w / 2 - 50, h / 2 - 50, 100, 100), 
		cv::Scalar(255, 255, 255, 255), CV_FILLED);
	cv::namedWindow("Modified Image", CV_WINDOW_AUTOSIZE);
	cv::imshow("Modified Image", img);
	std::string newEnd = "_mod.jpg";
	size_t posToReplace = path.find(".jpg");
	path.replace(posToReplace, newEnd.length(), newEnd);
	cv::imwrite(path, img);
	img.release();
	cv::waitKey();
	return 0;
}