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
	std::string pathx, pathy;
	char count[3];
	string window_name = "video | q or esc to quit";
	cout << "press space to save a picture. q or esc to quit" << endl;
	Mat next, prvs, frame, dst, flowx, flowy;
	double minf, maxf, mind, maxd;
	capture >> frame;
	if (frame.channels() == 3)
		cvtColor(frame, prvs, CV_BGR2GRAY);
	else 
		prvs = frame.clone();

	for (;;) {
		sprintf(count, "%03d", n);
		pathx = fn_out + '_' + count + "_x.mat";
		pathy = fn_out + '_' + count + "_y.mat";
		capture >> frame;
		if (frame.channels() == 3)
			cvtColor(frame, next, CV_BGR2GRAY);
		else 
			next = frame.clone();

		if (next.empty())
			break;
		processflow_gpu(prvs, next, flowx, flowy);
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
