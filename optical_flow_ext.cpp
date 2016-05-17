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

string type2str(int type) {
  string r;

  //printf("CV_MAT_DEPTH_MASK: %d\n", CV_MAT_DEPTH_MASK);
  //printf("CV_CN_SHIFT: %d\n", CV_CN_SHIFT);

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int writeMatToFile(const Mat &I, string path) {
 
    //load the matrix size
    int matWidth = I.size().width, matHeight = I.size().height;
    //read type from Mat
    int type = I.type();
    //declare values to be written
    float fvalue;
    double dvalue;
    Vec3f vfvalue;
    Vec3d vdvalue;
    //create the file stream
    ofstream file(path.c_str(), ios::out | ios::binary );
    if (!file)
        return -1;
    //write type and size of the matrix first
    file.write((const char*) &type, sizeof(type));
    file.write((const char*) &matWidth, sizeof(matWidth));
    file.write((const char*) &matHeight, sizeof(matHeight));
    //write data depending on the image's type
    switch (type)
    {
    default:
        cout << "Error: wrong Mat type: must be CV_32F, CV_64F, CV_32FC3 or CV_64FC3" << endl;
        break;
    // FLOAT ONE CHANNEL
    case CV_32FC1:
        cout << "Writing CV_32F image" << endl;
        for (int i=0; i < matWidth*matHeight; ++i) {
            fvalue = I.at<float>(i);
            file.write((const char*) &fvalue, sizeof(fvalue));
        }
        break;
    // DOUBLE ONE CHANNEL
    case CV_64FC1:
        cout << "Writing CV_64F image" << endl;
        for (int i=0; i < matWidth*matHeight; ++i) {
            dvalue = I.at<double>(i);
            file.write((const char*) &dvalue, sizeof(dvalue));
        }
        break;
    // FLOAT THREE CHANNELS
    case CV_32FC3:
        cout << "Writing CV_32FC3 image" << endl;
        for (int i=0; i < matWidth*matHeight; ++i) {
            vfvalue = I.at<Vec3f>(i);
            file.write((const char*) &vfvalue, sizeof(vfvalue));
        }
        break;
    // DOUBLE THREE CHANNELS
    case CV_64FC3:
        cout << "Writing CV_64FC3 image" << endl;
        for (int i=0; i < matWidth*matHeight; ++i) {
            vdvalue = I.at<Vec3d>(i);
            file.write((const char*) &vdvalue, sizeof(vdvalue));
        }
        break;
    }
    //close file
    file.close();
    return 0;
}
 
int readFileToMat(Mat &I, string path) {
    //declare image parameters
    int matWidth, matHeight, type;
    //declare values to be written
    float fvalue;
    double dvalue;
    Vec3f vfvalue;
    Vec3d vdvalue;
    //create the file stream
    ifstream file(path.c_str(), ios::in | ios::binary );
    if (!file)
        return -1;
    //read type and size of the matrix first
    file.read((char*) &type, sizeof(type));
    file.read((char*) &matWidth, sizeof(matWidth));
    file.read((char*) &matHeight, sizeof(matHeight));
    //change Mat type
    I = Mat::zeros(matHeight, matWidth, type);
    //write data depending on the image's type
    switch (type)
    {
    default:
        cout << "Error: wrong Mat type: must be CV_32F, CV_64F, CV_32FC3 or CV_64FC3" << endl;
        break;
    // FLOAT ONE CHANNEL
    case CV_32F:
        cout << "Reading CV_32F image" << endl;
        for (int i=0; i < matWidth*matHeight; ++i) {
            file.read((char*) &fvalue, sizeof(fvalue));
            I.at<float>(i) = fvalue;
        }
        break;
    // DOUBLE ONE CHANNEL
    case CV_64F:
        cout << "Reading CV_64F image" << endl;
        for (int i=0; i < matWidth*matHeight; ++i) {
            file.read((char*) &dvalue, sizeof(dvalue));
            I.at<double>(i) = dvalue;
        }
        break;
    // FLOAT THREE CHANNELS
    case CV_32FC3:
        cout << "Reading CV_32FC3 image" << endl;
        for (int i=0; i < matWidth*matHeight; ++i) {
            file.read((char*) &vfvalue, sizeof(vfvalue));
            I.at<Vec3f>(i) = vfvalue;
        }
        break;
    // DOUBLE THREE CHANNELS
    case CV_64FC3:
        cout << "Reading CV_64FC3 image" << endl;
        for (int i=0; i < matWidth*matHeight; ++i) {
            file.read((char*) &vdvalue, sizeof(vdvalue));
            I.at<Vec3d>(i) = vdvalue;
        }
        break;
    }
    //close file
    file.close();
    return 0;
}

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

int processflow_gpu(const Mat& frame0, const Mat& frame1, Mat& flowx, Mat& flowy, Mat& img)
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
	GpuMat planes[2];
	GpuMat dflowx, dflowy;
	cuda::split(d_flow, planes);
	dflowx = planes[0];
	dflowy = planes[1];
	dflowx.download(flowx);
	dflowy.download(flowy);

	return 0;
}

int process(VideoCapture& capture, std::string fn_out) {
	int n = 0;
	std::string pathx, pathy, path;
    path = fn_out + ".avi";
	char count[3];
	string window_name = "video | q or esc to quit";
	cout << "press space to save a picture. q or esc to quit" << endl;
	Mat next, prvs, frame, dst, flowx, flowy, flow, blend;
	double minf, maxf, mind, maxd;
	capture >> frame;
	if (frame.channels() == 3) {
		cvtColor(frame, prvs, CV_BGR2GRAY);
    }
	else {
		prvs = frame.clone();
    }

    int ex = CV_FOURCC('M', 'J', 'P', 'G');
    //int ex = static_cast<int>(capture.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
    Size S = Size((int) capture.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) capture.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter outputVideo;                                  // Open the output
    //outputVideo.open(path, 0, capture.get(CAP_PROP_FPS), S, true);
    //outputVideo.open(path, ex, capture.get(CAP_PROP_FPS), S, true);
    //outputVideo.open(path, CV_FOURCC('X','V','I','D'), capture.get(CAP_PROP_FPS), S, true);
    outputVideo.open(path, ex, 20, S, true);


	for (;;) {
		sprintf(count, "%03d", n);
		pathx = fn_out + '_' + count + "_x.mat";
		pathy = fn_out + '_' + count + "_y.mat";
		capture >> frame;
		if (frame.channels() == 3) {
			cvtColor(frame, next, CV_BGR2GRAY);
            //printf("converting frame to grayscale\n");
        }
		else {
			next = frame.clone();
        }

		if (next.empty())
			break;

        double minVal; 
        double maxVal; 
        Point minLoc; 
        Point maxLoc;
        cv::minMaxLoc( prvs, &minVal, &maxVal, &minLoc, &maxLoc );
        printf("max prvs value: %f\n", maxVal);
        cv::minMaxLoc( next, &minVal, &maxVal, &minLoc, &maxLoc );
        printf("max next value: %f\n", maxVal);

		processflow_gpu(prvs, next, flowx, flowy, flow);
        cv::addWeighted( frame, .7, flow, .3, 0.0, blend);


		string type;
		type = type2str(flowx.type());
		//printf("Matrix type: %s, cols: %d, rows: %d\n", type.c_str(), flowx.cols, flowx.rows);

		//delay N millis, usually long enough to display and capture input
		char key = (char)waitKey(30); 
		switch (key) {
		case 'q':
		case 'Q':
		case 27: //escape key
			return 0;
		default:
			break;
		}
        //outputVideo << dst;
        outputVideo.write(dst);
		writeMatToFile(flowx, pathx);
		writeMatToFile(flowy, pathy);
		prvs = next.clone();
		n++;
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
		 << "\texample: " << av[0] << " 0 output.avi" << endl
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
