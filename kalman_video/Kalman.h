#pragma once
#pragma once
#ifndef KALMAN_H
#define KALMAN_H

#include <iostream>
#include <cmath>


using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/video/video.hpp>

using namespace cv;


class Kalman
{
public:
	Kalman();
	VideoCapture capture;

	Mat frame2;
	Mat frame1;

	int k;

	const int HORIZONTAL_BORDER_CROP = 20;

	Mat smoothedMat;
	Mat affine;

	Mat smoothedFrame;

	double dx;
	double dy;
	double da;
	double ds_x;
	double ds_y;

	double sx;
	double sy;

	double scaleX;
	double scaleY;
	double thetha;
	double transX;
	double transY;

	double diff_scaleX;
	double diff_scaleY;
	double diff_transX;
	double diff_transY;
	double diff_thetha;

	double errscaleX;
	double errscaleY;
	double errthetha;
	double errtransX;
	double errtransY;

	double Q_scaleX;
	double Q_scaleY;
	double Q_thetha;
	double Q_transX;
	double Q_transY;

	double R_scaleX;
	double R_scaleY;
	double R_thetha;
	double R_transX;
	double R_transY;

	double sum_scaleX;
	double sum_scaleY;
	double sum_thetha;
	double sum_transX;
	double sum_transY;

	void Kalman_Filter(double *scaleX, double *scaleY, double *thetha, double *transX, double *transY);

	Mat filtered(Mat frame_1, Mat frame_2);

};

#endif //KALMAN_H