#!/usr/bin/env python
import sys, argparse 
from renderer import VideoStream, FlowStream

import pdb 

import numpy as np 
import scipy.io 

class Arguments:
	pass

args = Arguments()

args.fn_in='./video/20160412/stk_0001.avi'
args.flow_in='./video/20160412/stk_0001_flow/flow'
args.fn_out='./video/stack_0001.avi'
args.name='stack0001'

#Load flow data from directory
flowstream = FlowStream(args.flow_in)
ret_flow, flowframe = flowstream.peek()

averages = []
maximums = []
aveaccs = []
count = 0

prevframe = flowframe.copy()

while(flowstream.isOpened()):
	count += 1
	print 'Frame %d' % count 
	ret_flow, flowframe = flowstream.read()
	ave = np.mean(np.linalg.norm(flowframe, axis = 2))
	mx = np.max(np.linalg.norm(flowframe, axis = 2))
	aveacc = np.mean(np.linalg.norm(flowframe-prevframe, axis = 2))
	print "Average: %f, max: %f" % (ave, mx)
	aveaccs.append(aveacc)
	averages.append(ave)
	maximums.append(mx)
	prevframe = flowframe.copy()

aveaccs = np.array(aveaccs)
averages = np.array(averages)
maximums = np.array(maximums)

np.savez('./video/20160412/stk_0001.npz', averages = averages, maximums = maximums, aveacs = aveaccs)
scipy.io.savemat('./video/20160412/stk_0001.mat', {'averages': averages, 'maximums': maximums, 'aveacs': aveaccs})