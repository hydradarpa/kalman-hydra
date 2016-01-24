#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>

using namespace cv;


int main(int argc, char* argv[])
{
    // Load input video
    cv::VideoCapture input_cap("../video/local_prop_cb_with_bud.avi");
    if (!input_cap.isOpened())
    {
        std::cout << "!!! Input video could not be opened" << std::endl;
        return -1;
    }

    int codec = CV_FOURCC('M', 'J', 'P', 'G');
    // Setup output video
    //cv::VideoWriter output_cap("./output.avi",  
    //                           input_cap.get(CV_CAP_PROP_FOURCC),
    //                           input_cap.get(CV_CAP_PROP_FPS), 
    //                           cv::Size(input_cap.get(CV_CAP_PROP_FRAME_WIDTH), input_cap.get(CV_CAP_PROP_FRAME_HEIGHT)));

    cv::VideoWriter output_cap("./output.avi",  
                               codec,
                               20.0, 
                               cv::Size(input_cap.get(CV_CAP_PROP_FRAME_WIDTH), input_cap.get(CV_CAP_PROP_FRAME_HEIGHT)));

    if (!output_cap.isOpened())
    {
        std::cout << "!!! Output video could not be opened" << std::endl;
        return -1;
    }

    // Loop to read frames from the input capture and write it to the output capture
    cv::Mat frame;
    while (true)
    {       
        if (!input_cap.read(frame))             
            break;

        output_cap.write(frame);
    }

    // Release capture interfaces   
    input_cap.release();
    output_cap.release();

    return 0;
}