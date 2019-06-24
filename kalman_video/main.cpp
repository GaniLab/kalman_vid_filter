/* Real time video using kalman filter
 * Kalman parameters can be adjusted as needed
*/
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/tracking.hpp>

#include "Kalman.h"

using namespace std;
using namespace cv;


int main(int argc, char **argv)
{

	// create object for kalman class for refining video with kalman filter
	Kalman refining;

	// create object for capturing video with camera choice initialized at 0 (default camera)
	VideoCapture captureVideo(0);

	// create mat object to perform optical flow kalman filtering between current and next frame
	Mat frame_2, frame2;
	Mat frame_1, frame1;

	captureVideo >> frame_1;

	cvtColor(frame_1, frame1, COLOR_BGR2GRAY);

	int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');

	// save recorded video
	VideoWriter outputVideo;
	outputVideo.open("../video_output/after_refinement.avi", codec, 30, frame_1.size());

	// real time video capturing using frames
	while (true)
	{

		captureVideo >> frame_2;

		if (frame_2.data == NULL)
		{
			break;
		}

		cvtColor(frame_2, frame2, COLOR_BGR2GRAY);

		Mat smoothedFrame;

		smoothedFrame = refining.filtered(frame_1, frame_2);

		outputVideo.write(smoothedFrame);

		imshow("Refined Video", smoothedFrame);

		waitKey(10);

		// copy the frame
		frame_1 = frame_2.clone();
		frame2.copyTo(frame1);

	}

	return 0;
}
