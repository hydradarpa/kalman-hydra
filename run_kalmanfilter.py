#!/usr/bin/env python
import sys, argparse 
from kalman import KalmanFilter, IteratedMSKalmanFilter
from renderer import VideoStream, FlowStream
from distmesh_dyn import DistMesh

import logging

import pdb 

def main():
	usage = """run_kalmanfilter.py [input_avi_file] [optic_flow_path] [output_avi_file] [threshold]

HydraGL. State space model using an extended Kalman filter to track Hydra in video

Dependencies:
-Vispy*
-Numpy
-PyCuda**
-DistMesh 
-HDF
-OpenCV2
-matplotlib

Notes:
*  Uses OpenGL rendering. If using remotely, you'll need to set up a VirtualGL server
** If have a CUDA compatible graphics card

Example: 
./run_kalmanfilter.py ./video/johntest_brightcontrast_short.avi ... 
 ./video/johntest_brightcontrast_short/flow ./video/output.avi -s 15

For help:
./run_kalmanfilter.py -h 

Ben Lansdell
02/16/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('fn_in', default='./video/johntest_brightcontrast_short.avi', 
		help='input video file, any format readable by OpenCV', nargs = '?')
	parser.add_argument('flow_in', default='./video/johntest_brightcontrast_short/flow', 
		help='input optic flow path', nargs = '?')
	parser.add_argument('fn_out', default='./video/johntest_brightcontrast_short_output.avi', 
		help='avi output video file', nargs='?')
	parser.add_argument('-n', '--name', default='johntest_brightcontrast_short_lengthadapt', 
		help='name for saving run images', nargs='?')
	parser.add_argument('-t', '--threshold', default=9,
		help='threshold intensity below which is background', type = int)
	parser.add_argument('-s', '--gridsize', default=22,
		help='edge length for mesh (smaller is finer; unstable much further below 18)', type = int)
	parser.add_argument('-c', '--cuda', default=True,
		help='whether or not to do analysis on CUDA', type = bool)
	args = parser.parse_args()

	logging.basicConfig(filename='run_kalmanfilter.log',level=logging.DEBUG, \
		format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
	logging.info(' == run_kalmanfilter.py ==\nInitializing...')

	if len(sys.argv) == 1:
		print("No command line arguments provided, using defaults")
		logging.info("No command line arguments provided, using defaults")
	
	logging.info("Generating mesh")
	satisfied = False
	while not satisfied:
		capture = VideoStream(args.fn_in, args.threshold)
		frame = capture.current_frame()
		mask, ctrs, fd = capture.backsub()
		distmesh = DistMesh(frame, h0 = args.gridsize)
		distmesh.createMesh(ctrs, fd, frame, plot = True)
		a = raw_input('Is this mesh ok? Type ''thresh = xx'' to redo threshold and ''grid = xx'' to redo gridsize. Otherwise press ENTER: (currently threshold = %d, gridsize = %d) '%(args.threshold, args.gridsize))
		if len(a) == 0:
			satisfied = True
			break 
		words = a.split('=')
		if len(words) != 2:
			print 'Didn''t understand your response, continuing with current values'
			satisfied = True
		if words[0].strip().lower() == 'thresh':
			args.threshold = int(words[1])
		elif words[0].strip().lower() == 'grid':
			args.gridsize = int(words[1])
		else:
			print 'Didn''t understand your response, continuing with current values'
			satisfied = True

	logging.info("Created mesh")

	#Load flow data from directory
	flowstream = FlowStream(args.flow_in)
	ret_flow, flowframe = flowstream.peek()
	logging.info("Loaded optic flow data")

	if ret_flow:
		#alpha = 0.3
		kf = IteratedMSKalmanFilter(distmesh, frame, flowframe, cuda = args.cuda, sparse = True, multi = True, alpha = 0.3, nI = 4)
		#alpha = 0
		#kf = IteratedMSKalmanFilter(distmesh, frame, flowframe, cuda = args.cuda, sparse = True, multi = True, alpha = 0)
	else:
		print 'Cannot read flow stream'
		return 
	logging.info("Created Kalman Filter, Renderer and CUDA objects")

	#kf.compute(capture.gray_frame(), flowframe)
	logging.info("Starting main loop")
	nI = 3
	count = 0
	while(capture.isOpened()):
		count += 1
		print 'Frame %d' % count 
		ret, frame, grayframe, mask = capture.read()
		ret_flow, flowframe = flowstream.read()
		logging.info("Loaded frame %d"%count)

		if ret is False or ret_flow is False:
			break
		#for i in range(nI):
		#	print 'Iteration %d' % i 
		#	raw_input("Finished. Press Enter to continue")
		#	kf.compute(grayframe, flowframe)
		kf.compute(grayframe, flowframe, mask, imageoutput = 'screenshots/' + args.name + '_frame_%03d'%count)

	logging.info("Streams empty, closing")
	capture.release()
	output.release()
	cv2.destroyAllWindows()
	raw_input("Finished. Press ENTER to exit")

if __name__ == "__main__":
	sys.exit(main())