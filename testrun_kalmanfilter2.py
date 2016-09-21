#!/usr/bin/env python
import sys, argparse 
from kalman import KalmanFilter, IteratedMSKalmanFilter
from renderer import VideoStream, FlowStream
from distmesh_dyn import DistMesh

import pdb 

class Arguments:
	pass

args = Arguments()

args.fn_in='./video/20160412/stk_0001.avi'
args.flow_in='./video/20160412/stk_0001_flow/flow'
args.fn_out='./video/stack_0001.avi'
args.name='stack0001'
args.threshold=42
args.gridsize=25
args.cuda=True

nskip = 100
dm_out = './testrun_kalmanfilter_skip_%d.pkl' % nskip

capture = VideoStream(args.fn_in, args.threshold)

frame = capture.current_frame()
#mask, ctrs, fd = capture.backsub()

#Load flow data from directory
flowstream = FlowStream(args.flow_in)
ret_flow, flowframe = flowstream.peek()

for idx in range(nskip):
	ret, frame, grayframe, mask = capture.read()
	ret_flow, flowframe = flowstream.read()	

mask, ctrs, fd = capture.backsub()

#distmesh = DistMesh(frame, h0 = args.gridsize)
#distmesh.createMesh(ctrs, fd, frame, plot = True)
#Save this distmesh and reload it for quicker testing
#distmesh.save(dm_out)

#Load...
distmesh = DistMesh(frame, h0 = args.gridsize)
distmesh.load(dm_out)

if ret_flow:
	kf = IteratedMSKalmanFilter(distmesh, frame, flowframe, cuda = args.cuda, sparse = True, multi = True, alpha = 0.05, nI = 2)
else:
	print 'Cannot read flow stream'


#kf.compute(capture.gray_frame(), flowframe)
nI = 3
count = 0

#print 'Frame %d' % count 
#ret, frame, grayframe, mask = capture.read()
#ret_flow, flowframe = flowstream.read()

#kf.predict()
#kf.projectmask(mask)
#kf.state.refresh()

while(capture.isOpened()):
	count += 1
	print 'Frame %d' % count 
	ret, frame, grayframe, mask = capture.read()
	ret_flow, flowframe = flowstream.read()

	if ret is False or ret_flow is False:
		break
	#for i in range(nI):
	#	print 'Iteration %d' % i 
	#	raw_input("Finished. Press Enter to continue")
	#	kf.compute(grayframe, flowframe)
	kf.compute(grayframe, flowframe, mask, imageoutput = 'screenshots/' + args.name + '_frame_' + str(count))

capture.release()
output.release()
cv2.destroyAllWindows()
raw_input("Finished. Press ENTER to exit")
