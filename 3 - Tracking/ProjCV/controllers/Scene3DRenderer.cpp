/*
 * Scene3DRenderer.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Scene3DRenderer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/video/background_segm.hpp>
#include <stddef.h>
#include <string>

#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Scene properties class (mostly called by Glut)
 */
Scene3DRenderer::Scene3DRenderer(
		Reconstructor &r, const vector<Camera*> &cs) :
				m_reconstructor(r),
				m_cameras(cs),
				m_num(4),
				m_sphere_radius(1850)
{
	m_width = 1024;
	m_height = 1000;
	m_quit = false;
	m_paused = false;
	m_rotate = false;
	m_camera_view = true;
	m_show_volume = true;
	m_show_grd_flr = true;
	m_show_cam = true;
	m_show_org = true;
	m_show_arcball = false;
	m_show_info = true;
	m_fullscreen = false;

	// Read the checkerboard properties (XML)
	FileStorage fs;
	fs.open(m_cameras.front()->getDataPath() + ".." + string(PATH_SEP) + General::CBConfigFile, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["CheckerBoardWidth"] >> m_board_size.width;
		fs["CheckerBoardHeight"] >> m_board_size.height;
		fs["CheckerBoardSquareSize"] >> m_square_side_len;
	}
	fs.release();

	m_current_camera = 0;
	m_previous_camera = 0;

	m_number_of_frames = m_cameras.front()->getFramesAmount();
	m_current_frame = 0;
	m_previous_frame = -1;

	const int H = 0;
	const int S = 0;
	const int V = 0;
	m_h_threshold = H;
	m_ph_threshold = H;
	m_s_threshold = S;
	m_ps_threshold = S;
	m_v_threshold = V;
	m_pv_threshold = V;

	createTrackbar("Frame", VIDEO_WINDOW, &m_current_frame, m_number_of_frames - 2);
	//createTrackbar("H", VIDEO_WINDOW, &m_h_threshold, 255);
	//createTrackbar("S", VIDEO_WINDOW, &m_s_threshold, 255);
	//createTrackbar("V", VIDEO_WINDOW, &m_v_threshold, 255);




	createFloorGrid();
	setTopView();
}

/**
 * Deconstructor
 * Free the memory of the floor_grid pointer vector
 */
Scene3DRenderer::~Scene3DRenderer()
{
	for (size_t f = 0; f < m_floor_grid.size(); ++f)
		for (size_t g = 0; g < m_floor_grid[f].size(); ++g)
			delete m_floor_grid[f][g];
}

/**
 * Process the current frame on each camera
 */
bool Scene3DRenderer::processFrame()
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_current_frame == m_previous_frame + 1)
		{
			m_cameras[c]->advanceVideoFrame();
		}
		else if (m_current_frame != m_previous_frame)
		{
			m_cameras[c]->getVideoFrame(m_current_frame);
		}
		assert(m_cameras[c] != nullptr);
		processForeground(m_cameras[c]);

		
	}
	return true;
}

void CalcDiffImage(const Mat &a, const Mat &b, Mat &out) {
	Vec3b *data_a = (Vec3b *)a.data;
	Vec3b *data_b = (Vec3b *)b.data;
	uchar *data_out = out.data;
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < out.total(); ++i) {
		Vec3i diff = (Vec3i)data_a[i] - (Vec3i)data_b[i];
		double Sa = (double)data_a[i][1] / 255.0;
		double Sb = (double)data_b[i][1] / 255.0;
		double Ha = (double)data_a[i][0] / 180.0;
		double Hb = (double)data_b[i][0] / 180.0;
		double scaler = pow(min((min(Sa, Sb) * abs(Ha - Hb) + abs(Sa - Sb)), 1), 0.3);
		data_out[i] = sqrt(pow(diff[2], 2) + pow(diff[1], 2) + pow(diff[0], 2)) * scaler;
	}
}

/**
 * Separate the background from the foreground
 * ie.: Create an 8 bit image where only the foreground of the scene is white (255)
 */
void Scene3DRenderer::processForeground(
		Camera* camera)
{
	assert(!camera->getFrame().empty());
	Mat hsv_image;
	cvtColor(camera->getFrame(), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	Mat back_image;
	merge(camera->getBgHsvChannels(), back_image);

	Mat diff(back_image.rows, back_image.cols, CV_8UC1);
	CalcDiffImage(hsv_image, back_image, diff);

	Mat foreground;
	double tresh = threshold(diff, foreground, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	threshold(diff, foreground, tresh * 0.5, 255, CV_THRESH_BINARY);
	bitwise_or(foreground, diff, foreground);
	threshold(foreground, foreground, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	// Improve the foreground image
	int dilate_type = MORPH_ELLIPSE;
	int dilate_size = 2;
	int erode_type = MORPH_ELLIPSE;
	int erode_size = 2;

	Mat dilate_element = getStructuringElement(dilate_type,
		Size(2 * dilate_size + 1, 2 * dilate_size + 1),
		Point(dilate_size, dilate_size));

	Mat erode_element = getStructuringElement(erode_type,
		Size(2 * erode_size + 1, 2 * erode_size + 1),
		Point(erode_size, dilate_size));
	
	dilate(foreground, foreground, dilate_element);
	erode(foreground, foreground, erode_element);
	medianBlur(foreground, foreground, 5);

	camera->setForegroundImage(foreground);
}

/**
 * Set currently visible camera to the given camera id
 */
void Scene3DRenderer::setCamera(
		int camera)
{
	m_camera_view = true;

	if (m_current_camera != camera)
	{
		m_previous_camera = m_current_camera;
		m_current_camera = camera;
		m_arcball_eye.x = m_cameras[camera]->getCameraPlane()[0].x;
		m_arcball_eye.y = m_cameras[camera]->getCameraPlane()[0].y;
		m_arcball_eye.z = m_cameras[camera]->getCameraPlane()[0].z;
		m_arcball_up.x = 0.0f;
		m_arcball_up.y = 0.0f;
		m_arcball_up.z = 1.0f;
	}
}

/**
 * Set the 3D scene to bird's eye view
 */
void Scene3DRenderer::setTopView()
{
	m_camera_view = false;
	if (m_current_camera != -1)
		m_previous_camera = m_current_camera;
	m_current_camera = -1;

	m_arcball_eye = vec(0.0f, 0.0f, 10000.0f);
	m_arcball_centre = vec(0.0f, 0.0f, 0.0f);
	m_arcball_up = vec(0.0f, 1.0f, 0.0f);
}

/**
 * Create a LUT for the floor grid
 */
void Scene3DRenderer::createFloorGrid()
{
	const int size = m_reconstructor.getSize() / m_num;
	const int z_offset = 3;

	// edge 1
	vector<Point3i*> edge1;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge1.push_back(new Point3i(-size * m_num, y, z_offset));

	// edge 2
	vector<Point3i*> edge2;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge2.push_back(new Point3i(x, size * m_num, z_offset));

	// edge 3
	vector<Point3i*> edge3;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge3.push_back(new Point3i(size * m_num, y, z_offset));

	// edge 4
	vector<Point3i*> edge4;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge4.push_back(new Point3i(x, -size * m_num, z_offset));

	m_floor_grid.push_back(edge1);
	m_floor_grid.push_back(edge2);
	m_floor_grid.push_back(edge3);
	m_floor_grid.push_back(edge4);
}

} /* namespace nl_uu_science_gmt */
