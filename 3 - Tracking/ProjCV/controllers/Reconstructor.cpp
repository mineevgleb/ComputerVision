/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <cassert>
#include <iostream>
#include <set>

#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
	Reconstructor::Reconstructor(
		const vector<Camera*> &cs) :
		m_cameras(cs),
		m_height(2048),
		m_edge(6656),
		m_step(32),
		m_clusterCenters(4),
		m_clustered(false)
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();
	}

	m_voxels_amount = (m_edge / m_step) * (m_edge / m_step) * (m_height / m_step);

	initialize();
}

/**
 * Deconstructor
 * Free the memory of the pointer vectors
 */
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */
void Reconstructor::initialize()
{
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_edge / 2;
	const int xR = m_edge / 2;
	const int yL = -m_edge / 2;
	const int yR = m_edge / 2;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels ";
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			cout << done << "%..." << flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}
	cout << "done!" << endl;
}

void Reconstructor::markClusters(bool updateExisting) {
	std::vector<cv::Point2f> points;
	std::vector<int> labels(m_visible_voxels.size());
	cv::Mat centers;
	for (int i = 0; i < m_visible_voxels.size(); ++i) {
		points.push_back(cv::Point(m_visible_voxels[i]->x, m_visible_voxels[i]->y));
		if (updateExisting) {
			labels[i] = m_visible_voxels[i]->label;
		}
	}
	if (!updateExisting) {
		cv::kmeans(points, 4, labels, cv::TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1.0), 4, KMEANS_PP_CENTERS, centers);
		for (int i = 0; i < m_visible_voxels.size(); i++) {
			float dist = norm(Point(m_visible_voxels[i]->x, m_visible_voxels[i]->y) -
				Point(centers.at<float>(labels[i], 0), centers.at<float>(labels[i], 1)));
				if (dist < 600) {
					m_visible_voxels[i]->label = labels[i];
				}
				else {
					m_visible_voxels[i] = *m_visible_voxels.rbegin();
					m_visible_voxels.pop_back();
				}
		}
		int lbl[4];
		DetermineLabels(lbl);
		for (int i = 0; i < m_visible_voxels.size(); i++) {
			m_visible_voxels[i]->label = labels[i] = lbl[m_visible_voxels[i]->label];
		}
		cv::kmeans(points, 4, labels, cv::TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1.0), 1, KMEANS_USE_INITIAL_LABELS, centers);
		for (int i = 0; i < 4; ++i) {
			m_clusterCenters[i] = cv::Point2i(centers.at<float>(i, 0), centers.at<float>(i, 1));
		}
		m_clustered = true;
	}
	else {
		cv::kmeans(points, 4, labels, cv::TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1.0), 1, KMEANS_USE_INITIAL_LABELS, centers);
		for (int i = 0; i < 4; ++i) {
			m_clusterCenters[i] = cv::Point2i(centers.at<float>(i, 0), centers.at<float>(i, 1));
		}
	}
	m_centersTracks.push_back(m_clusterCenters);
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	m_visible_voxels.clear();
	std::vector<Voxel*> visible_voxels;

	int v;
#pragma omp parallel for private(v) shared(visible_voxels)
	for (v = 0; v < (int) m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
			int minLabel = 0;
			float minDist = norm(m_clusterCenters[0] - Point2f(voxel->x, voxel->y));
			for (int i = 1; i < 4; ++i) {
				float dist = norm(m_clusterCenters[i] - Point2f(voxel->x, voxel->y));
				if (dist < minDist) {
					minLabel = i;
					minDist = dist;
				}
			}
			voxel->label = minLabel;
#pragma omp critical //push_back is critical
			if (minDist < 600 || !m_clustered)
				visible_voxels.push_back(voxel);
		}
	}
	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());

	//Build surface
	int sizeh = m_height / m_step;
	int sizew = m_edge / m_step;
	PolyVox::SimpleVolume<uint8_t> volData(PolyVox::Region(PolyVox::Vector3DInt32(0, 0, 0), PolyVox::Vector3DInt32(sizew, sizew, sizeh)));

	for (size_t v = 0; v < visible_voxels.size(); v++)
	{
		volData.setVoxelAt(visible_voxels[v]->x / m_step + sizew / 2, visible_voxels[v]->y / m_step + sizew / 2, visible_voxels[v]->z / m_step + 1, 255);
	}
	PolyVox::MarchingCubesSurfaceExtractor<PolyVox::SimpleVolume<uint8_t>> surfaceExtractor(&volData, volData.getEnclosingRegion(), &m_mesh);
	surfaceExtractor.execute();
}

void Reconstructor::CreateColorModels(HistModel *models) const {
	//for each camera
	for (int i = 0; i < 4; ++i) {
		Mat labelMask(m_cameras[0]->getSize(), CV_8U, cv::Scalar::all(0));
		Mat zBuffer(m_cameras[0]->getSize(), CV_32F, 0.0f);
		for (Voxel *v : m_visible_voxels) {
			Point proj = v->camera_projection[i];
			uchar voxLabel = v->label;
			uchar maskLabel = labelMask.at<uchar>(proj);
			float dist = norm(m_cameras[i]->getCameraLocation() - Point3f(v->x, v->y, v->z));
			if (maskLabel == 0 || dist < zBuffer.at<float>(proj)) {
				circle(zBuffer, proj, 3, Scalar(dist), -1);
				circle(labelMask, proj, 3, Scalar(voxLabel + 1), -1);
			}
		}
		Mat foreg = m_cameras[i]->getFrame();
		for (int j = 0; j < foreg.rows; ++j) {
			for (int k = 0; k < foreg.cols; ++k) {
				char label = labelMask.at<uchar>(j, k);
				if (label != 0) {
					Vec3b color = foreg.at<Vec3b>(j, k);
					models[label - 1].AddPt(color[0], color[1], color[2]);
				}
			}
		}
	}
}

void Reconstructor::SaveColorModels() const {
	HistModel models[4];
	CreateColorModels(models);
	models[0].Save("person1cm.txt");
	models[1].Save("person2cm.txt");
	models[2].Save("person3cm.txt");
	models[3].Save("person4cm.txt");
}

void Reconstructor::DetermineLabels(int *labels) const {
	HistModel modelsCur[4];
	CreateColorModels(modelsCur);
	HistModel modelsOrig[4];
	modelsOrig[0].Load("person1cm.txt");
	modelsOrig[1].Load("person2cm.txt");
	modelsOrig[2].Load("person3cm.txt");
	modelsOrig[3].Load("person4cm.txt");
	std::set<int> freeLabels = { 0, 1, 2, 3 };
	int res[4];
	float minErr = 1000.0;
	for (int i : freeLabels) {
		res[0] = i;
		std::set<int> freeLabels2 = freeLabels;
		freeLabels2.erase(i);
		for (int j : freeLabels2) {
			res[1] = j;
			std::set<int> freeLabels3 = freeLabels2;
			freeLabels3.erase(j);
			for (int k : freeLabels3) {
				res[2] = k;
				std::set<int> freeLabels4 = freeLabels3;
				freeLabels4.erase(k);
				for (int m : freeLabels4) {
					res[3] = m;
					float err = modelsCur[0].CalcDiff(&modelsOrig[res[0]]);
					err += modelsCur[1].CalcDiff(&modelsOrig[res[1]]);
					err += modelsCur[2].CalcDiff(&modelsOrig[res[2]]);
					err += modelsCur[3].CalcDiff(&modelsOrig[res[3]]);
					if (err < minErr) {
						minErr = err;
						for (int v = 0; v < 4; ++v) {
							labels[v] = res[v];
						}
					}
				}
			}
		}
	}
}

} /* namespace nl_uu_science_gmt */
