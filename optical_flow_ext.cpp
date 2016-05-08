#include <iostream>
#include <fstream>
#include <stdio.h>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

inline bool isFlowCorrect(Point2f u)
{
	return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

int processflow_gpu(const Mat& frame0, const Mat& frame1, Mat& flowx, Mat& flowy)
{
	GpuMat d_frame0(frame0);
	GpuMat d_frame1(frame1);
	GpuMat d_flow(frame0.size(), CV_32FC2);
	Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
	{
		GpuMat d_frame0f;
		GpuMat d_frame1f;
		d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
		d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
		const int64 start = getTickCount();
		brox->calc(d_frame0f, d_frame1f, d_flow);
		const double timeSec = (getTickCount() - start) / getTickFrequency();
		cout << "Brox : " << timeSec << " sec" << endl;
	}
	GpuMat planes[2];
	cuda::split(d_flow, planes);
	//Mat flowx(planes[0]);
	//Mat flowy(planes[1]);
	flowx = planes[0];
	flowy = planes[1];
	return 0;
}

int process(VideoCapture& capture, std::string fn_out) {
	int n = 0;
	char filename[200];
	string window_name = "video | q or esc to quit";
	cout << "press space to save a picture. q or esc to quit" << endl;
	Mat next, prvs, frame, flowx, flowy, dst;
	double minf, maxf, mind, maxd;
	capture >> frame;
	cvtColor(frame, prvs, CV_BGR2GRAY);
	for (;;) {
		capture >> frame;
		cvtColor(frame, next, CV_BGR2GRAY);
		if (next.empty())
			break;
		processflow_gpu(prvs, next, flowx, flowy);
		//delay N millis, usually long enough to display and capture input
		char key = (char)waitKey(30); 
		switch (key) {
		case 'q':
		case 'Q':
		case 27: //escape key
			return 0;
		case ' ': //Save an image
			sprintf(filename,"filename_x_%.3d.ext",n++);
			imwrite(filename,flowx);
			//sprintf(filename,"filename_y_%.3d.ext",n++);
			//imwrite(filename,flowy);
			cout << "Saved " << filename << endl;
			break;
		default:
			break;
		}
		prvs = next.clone();
	}
	cout << "Finished. ENTER to exit" << endl;
	getchar();
	return 0;
}

void help(char** av) {
	cout << "This program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer." << endl
		 << "It then uses a Brox GPU implementation to compute optic flow" << endl
		 << endl
		 << "Usage:\n" << av[0] << " <video file, image sequence or device number> <output filename>" << endl
		 << "q,Q,esc -- quit" << endl
		 << "space   -- save frame" << endl << endl
		 << "\tTo capture from a camera pass the device number. To find the device number, try ls /dev/video*" << endl
		 << "\texample: " << av[0] << " 0" << endl
		 << "\tYou may also pass a video file instead of a device number" << endl
		 << "\texample: " << av[0] << " video.avi" << endl
		 << "\tYou can also pass the path to an image sequence and OpenCV will treat the sequence just like a video." << endl
		 << "\texample: " << av[0] << " right%%02d.jpg" << endl;
}

int main(int ac, char** av) {
	if (ac != 3) {
		help(av);
		return 1;
	}
	std::string arg = av[1];
	std::string fn_out = av[2];
	VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file or image sequence
	if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
		capture.open(atoi(arg.c_str()));
	if (!capture.isOpened()) {
		cerr << "Failed to open the video device, video file or image sequence!\n" << endl;
		help(av);
		return 1;
	}
	return process(capture, fn_out);
}