#!/usr/bin/env python
import sys 
from kalman import KalmanFilter
from renderer import VideoStream
from distmesh_dyn import DistMesh

def main():
	    usage = """HydraGL

State space model using an extended Kalman filter to track Hydra in video

Usage:
	run_kalmanfilter.py [input_avi_file] [output_avi_file] [threshold]

Input:
	input_avi_file = (optional, default = ./video/GCaMP_local_prop.avi)
	output_avi_file = (optional, default = ./video/GCaMP_local_kalman.avi)
	threshold = (optional, default = 9) threshold intensity below which is background 

Dependencies:
-Vispy*
-Numpy
-PyCuda**
-DistMesh 
-HDF
-OpenCV2

Notes:
*  Uses OpenGL rendering. If using remotely, you'll need to set up a VirtualGL
   server
** Requires a CUDA compatible graphics card

Ben Lansdell
02/16/2016
"""

    fmt = optparse.IndentedHelpFormatter(max_help_position=50, width=100)
    parser = optparse.OptionParser(usage=usage, formatter=fmt)
    group = optparse.OptionGroup(parser, 'Input data',
                                 'Input parameters.')
    group.add_option('-t', '--threshold', default=9, dest='threshold'
                     help='Threshold intensity below which is background')
    group.add_option('-i', '--input', metavar='WORDS', 
    	default='./video/GCaMP_local_prop.avi', dest='fn_in', help='avi input video file')
    group.add_option('-o', '--output', metavar='WORDS', dest='fn_out',
    	default='./video/GCaMP_local_prop_kalman.avi', help='avi output video file')
    parser.add_option_group(group)
	
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
	
	options, _ = parser.parse_args()
	print options.fn_in, options.fn_out, options.threshold 

	capture = VideoStream(options.fn_in, options.threshold)
	frame = capture.current_frame()
	mask, ctrs, fd = capture.backsub()
	distmesh = DistMesh(frame)
	distmesh.createMesh(ctrs, fd, frame, True)
	
	#Load flow data from .hdf file
	#flowframe = capture.backsub(hdf.read())
	
	#kf = KalmanFilter(distmesh, frame)
	#kf.compute(capture.gray_frame(), flowframe)
	
	#while(capture.isOpened()):
	#	ret, frame, grayframe = capture.read(backsub = True)
	#	flowframe = capture.backsub(hdf.read())
	#	if ret == False:
	#		break 
	#	kf.compute(grayframe, flowframe)
	
	#capture.release()
	#output.release()
	#cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())