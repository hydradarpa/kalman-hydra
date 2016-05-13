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

static Vec3b computeColor(float fx, float fy)
{
	static bool first = true;

	// relative lengths of color transitions:
	// these are chosen based on perceptual similarity
	// (e.g. one can distinguish more shades between red and yellow
	//  than between yellow and green)
	const int RY = 15;
	const int YG = 6;
	const int GC = 4;
	const int CB = 11;
	const int BM = 13;
	const int MR = 6;
	const int NCOLS = RY + YG + GC + CB + BM + MR;
	static Vec3i colorWheel[NCOLS];

	if (first)
	{
		int k = 0;

		for (int i = 0; i < RY; ++i, ++k)
			colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

		for (int i = 0; i < YG; ++i, ++k)
			colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

		for (int i = 0; i < GC; ++i, ++k)
			colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

		for (int i = 0; i < CB; ++i, ++k)
			colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

		for (int i = 0; i < BM; ++i, ++k)
			colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

		for (int i = 0; i < MR; ++i, ++k)
			colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

		first = false;
	}

	const float rad = sqrt(fx * fx + fy * fy);
	const float a = atan2(-fy, -fx) / (float) CV_PI;

	const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
	const int k0 = static_cast<int>(fk);
	const int k1 = (k0 + 1) % NCOLS;
	const float f = fk - k0;

	Vec3b pix;

	for (int b = 0; b < 3; b++)
	{
		const float col0 = colorWheel[k0][b] / 255.0f;
		const float col1 = colorWheel[k1][b] / 255.0f;

		float col = (1 - f) * col0 + f * col1;

		if (rad <= 1)
			//col = 1 - rad * (1 - col); // increase saturation with radius
			col = rad * (col); // increase saturation with radius
		else
			col *= .75; // out of range

		pix[2 - b] = static_cast<uchar>(255.0 * col);
	}

	return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
	dst.create(flowx.size(), CV_8UC3);
	dst.setTo(Scalar::all(0));

	// determine motion range:
	float maxrad = maxmotion;

	if (maxmotion <= 0)
	{
		maxrad = 1;
		for (int y = 0; y < flowx.rows; ++y)
		{
			for (int x = 0; x < flowx.cols; ++x)
			{
				Point2f u(flowx(y, x), flowy(y, x));

				if (!isFlowCorrect(u))
					continue;

				maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
			}
		}
	}

	for (int y = 0; y < flowx.rows; ++y)
	{
		for (int x = 0; x < flowx.cols; ++x)
		{
			Point2f u(flowx(y, x), flowy(y, x));

			if (isFlowCorrect(u))
				dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
		}
	}
}

static void showFlow(const char* name, const GpuMat& d_flow, Mat& img)
{
	GpuMat planes[2];
	cuda::split(d_flow, planes);
	Mat flowx(planes[0]);
	Mat flowy(planes[1]);
	drawOpticalFlow(flowx, flowy, img, 10);
}

int processflow_gpu(const Mat& frame0, const Mat& frame1, Mat& img)
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
		showFlow("Brox", d_flow, img);
	}
	return 0;
}

int process(VideoCapture& capture, std::string fn_out) {
	int n = 0;
	char filename[200];
	string window_name = "video | q or esc to quit";
	cout << "press space to save a picture. q or esc to quit" << endl;
	//namedWindow(window_name, WINDOW_KEEPRATIO); //resizable window;
	Mat next, prvs, frame, flow, dst;
	double minf, maxf, mind, maxd;
	capture >> frame;
	cvtColor(frame, prvs, CV_BGR2GRAY);
	//prvs = frame.clone();
    int ex = CV_FOURCC('M', 'J', 'P', 'G');
	//int ex = static_cast<int>(capture.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
    Size S = Size((int) capture.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) capture.get(CAP_PROP_FRAME_HEIGHT));
	VideoWriter outputVideo;                                  // Open the output
	//outputVideo.open(fn_out, 0, capture.get(CAP_PROP_FPS), S, true);
	//outputVideo.open(fn_out, ex, capture.get(CAP_PROP_FPS), S, true);
	//outputVideo.open(fn_out, CV_FOURCC('X','V','I','D'), capture.get(CAP_PROP_FPS), S, true);
	outputVideo.open(fn_out, ex, 20, S, true);

	for (;;) {
		capture >> frame;
		cvtColor(frame, next, CV_BGR2GRAY);
		//next = frame.clone();
		sprintf(filename,"filename%.3d.jpg",n++);
		//imwrite(filename,dst);
		imwrite(filename,frame);
	
		//printf("Hi");
		//if (next.empty())
		//	printf("Hi");
		//	break;
		//Process flow
		processflow_gpu(prvs, next, flow);
		cv::addWeighted( frame, .7, flow, .3, 0.0, dst);
		//cv::add( frame, flow, dst);
		//imshow(window_name, dst);
		char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input
		switch (key) {
		case 'q':
		case 'Q':
		case 27: //escape key
			return 0;
		case ' ': //Save an image
			sprintf(filename,"filename%.3d.jpg",n++);
			//imwrite(filename,dst);
			imwrite(filename,flow);
			cout << "Saved " << filename << endl;
			break;
		default:
			break;
		}
		//outputVideo << dst;
		//outputVideo.write(dst);
		outputVideo.write(flow);
		prvs = next.clone();
	}
	
	cout << "Finished. ENTER to exit" << endl;
	outputVideo.release();
	//getchar();
	return 0;
}

void help(char** av) {
	cout << "This program captures frames from a video file, image sequence (01.jpg, 02.jpg ... 10.jpg) or camera connected to your computer." << endl
		 << "It then uses a Brox GPU implementation to compute optic flow" << endl
		 << endl
		 << "Usage:\n" << av[0] << " <video file, image sequence or device number> <output filename>" << endl
		 << "q,Q,esc -- quit" << endl
		 << "space   -- save frame" << endl << endl
		 << "\tYou may also pass a video file instead of a device number" << endl
		 << "\texample: " << av[0] << " video.avi output.avi" << endl
		 << "\tYou can also pass the path to an image sequence and OpenCV will treat the sequence just like a video." << endl
		 << "\texample: " << av[0] << " right%%02d.jpg output.avi" << endl;
}

int main(int ac, char** av) {
	if (ac != 3) {
		help(av);
		return 1;
	}

	std::string arg = av[1];
	//std::string arg = "000.png";
	std::string fn_out = av[2];
	//std::string pathToData("000.png");

	VideoCapture capture(arg); //try to open string, this will attempt to open it as a video file or image sequence
	//VideoCapture capture(pathToData); //try to open string, this will attempt to open it as a video file or image sequence
	if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
		capture.open(atoi(arg.c_str()));
		cout << "Failed to create on first try. Trying using integer argument" << endl;
	if (!capture.isOpened()) {
		cerr << "Failed to open the video device, video file or image sequence!\n" << endl;
		help(av);
		return 1;
	}
	return process(capture, fn_out);
}