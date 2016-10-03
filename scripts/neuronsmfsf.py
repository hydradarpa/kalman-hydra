#!/usr/bin/env python
import sys, os
from renderer import VideoStream
from imgproc import load_ground_truth
from scipy.io import loadmat 

import cv2 
import numpy as np 
from matplotlib import pyplot as plt
import matplotlib.patches as patches

fn_in='./video/20160412/stk_0001.avi'
name='stack0001'

mfsf_in = './mfsf_results/stk_0001_mfsf_nref100/'
groundtruth_in = './analysis/stack_0001-1-620-groundtruthneurontracks.csv'

imageoutput = mfsf_in + '/tracking/'

threshold = 1
tracked_thresh = 6
corr_window = 30

#Make directory if needed...
if not os.path.exists(imageoutput):
	os.makedirs(imageoutput)

#Load MFSF data
a = loadmat(mfsf_in + '/result.mat')

params = a['parmsOF']
u = a['u']
v = a['v']

#Find reference frame 
nref = params['nref'][0,0][0,0]

#Skip to this frame and create mesh 
capture = VideoStream(fn_in, threshold)

nx = int(capture.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
nF = int(capture.cap.get(cv2.CAP_PROP_FRAME_COUNT))

nF = u.shape[2]

#Copy ground truth tracking structure for flow tracking estimates
true_cells = load_ground_truth(groundtruth_in)
estimate_cells = [dict() for _ in range(nF+1)]

nC = len(true_cells[0][0].keys())

#Initialize with true locations
truepositions = np.zeros((nF, nC, 2))
estpositions = np.zeros((nF, nC, 2))
refpositions = np.zeros((nC, 2))
distance_error = np.zeros((nF, nC))
max_error = np.zeros(nC)
trackedprop = np.zeros(nF)
corr_windowed = np.zeros((nC, nF))
colors = np.zeros((nF, nC))

for frame in range(nF):
	for cell, loc in true_cells[0][frame].iteritems():
		truepositions[frame,cell-1,:] = loc 

for cell, loc in true_cells[0][nref].iteritems():
	refpositions[cell-1,:] = loc 

for idx in range(nF):
	#Update tracked positions
	dx = u[refpositions[:,1].astype(int), refpositions[:,0].astype(int), idx]
	dy = v[refpositions[:,1].astype(int), refpositions[:,0].astype(int), idx]
	estpositions[idx,:,:] = refpositions.copy()
	estpositions[idx,:,0] += dx
	estpositions[idx,:,1] += dy
	for c in range(nC):
		tp = truepositions[idx,c,:]
		ep = estpositions[idx,c,:]
		distance_error[idx,c] = np.sqrt(np.sum((tp-ep)*(tp-ep)))
	trackedprop[idx] = 100*np.sum(distance_error[idx,:] < tracked_thresh)/nC

tracked = (1+np.tanh((distance_error-10)/1.5))/2
colors = np.zeros((nF, nC, 3))
colors[:,:,0] = tracked
colors[:,:,1] = 1-tracked

#For each cell compute the correlation between its labeled activity and its 'true activity'
#over a window

#Make cell specific tracking directories
for c in range(nC):
	#Compute max error distance 
	max_error[c] = np.max(distance_error[:,c])
	celloutput = imageoutput + '/celltrack_maxerror_%03d_cell_%03d/'%(int(max_error[c]), c)
	if not os.path.exists(celloutput):
		os.makedirs(celloutput)
	#for idx in range(nF):

tracked_int = np.zeros(corr_window)
true_int = np.zeros(corr_window)

for idx in range(nF):
	print("Visualizing frame %d" % idx)
	fn_out = imageoutput + "frame_%03d.png"%idx
	#Read in frame
	ret, frame, grayframe, mask = capture.read()

	#Render current frame
	plt.clf()
	img_plot = plt.imshow(frame)

	#Draw ground truth in yellow
	plt.scatter(x = truepositions[idx,:,0], y = truepositions[idx,:,1], color = 'b', s = 1)

	#Draw tracking in red/green depending on distance (green points are smaller, 
	#	red (errors) are larger)
	plt.scatter(x = estpositions[idx,:,0], y = estpositions[idx,:,1], color=colors[idx,:,:], s = 1)
	plt.axis('off')

	axes = plt.gca()
	axes.set_xlim([0, nx])
	axes.set_ylim([0, nx])
	axes.invert_yaxis()
	plt.savefig(fn_out, bbox_inches = 'tight')

	#Sub window with larger frame and box highlighting where we are
	subplot = plt.axes([.65, .7, .2, .2], axisbg='y')
	plt.imshow(frame)

	#Plot also neuron-centric tracking
	c = 0
	true_pos = np.squeeze(truepositions[idx,c,:])
	pred_pos = np.squeeze(estpositions[idx,c,:])
	txt = axes.text(0, 0, 'Error: %f'%distance_error[idx,c], color = (1, 0.3, 0.3))
	plt.sca(axes)
	s1 = plt.scatter(x = truepositions[idx,c,0], y = truepositions[idx,c,1], color = (0, 0, 1), s = 10, marker='o')	
	s2 = plt.scatter(x = estpositions[idx,c,0], y = estpositions[idx,c,1], color=colors[idx,c,:], s = 10, marker='o')
	#Plot line between two points when separated by large amount
	l = plt.plot([pred_pos[0], true_pos[0]], [pred_pos[1], true_pos[1]], color='r', linestyle = 'dashed')

	rect = subplot.add_patch(
		patches.Rectangle(
			(0, 0),
			10,
			10,
			fill=False, edgecolor = (1,0.3,0.3)
		)
	)

	for c in range(nC):
		print 'Frame: %d, Cell: %d'%(idx,c)
		celloutput = imageoutput + '/celltrack_maxerror_%03d_cell_%03d/'%(int(max_error[c]), c)
		fn_out = celloutput + '/frame_%03d.png'%idx
		#plt.clf()
		#Get tracked and true neuron locations
		true_pos = np.squeeze(truepositions[idx,c,:])
		pred_pos = np.squeeze(estpositions[idx,c,:])

		#Decide on window to plot
		pminx = min(true_pos[0], pred_pos[0])
		pminy = min(true_pos[1], pred_pos[1])
		pmaxx = max(true_pos[0], pred_pos[0])
		pmaxy = max(true_pos[1], pred_pos[1])

		xmin = pminx - max(25, 1.5*(pmaxx - pminx))
		xmax = pmaxx + max(25, 1.5*(pmaxx - pminx))
		ymin = pminy - max(25, 1.5*(pmaxy - pminy))
		ymax = pmaxy + max(25, 1.5*(pmaxy - pminy))

		#Make square
		if xmax-xmin > ymax-ymin:
			ymin = (pminy + pmaxy)/2 - (xmax-xmin)/2
			ymax = (pminy + pmaxy)/2 + (xmax-xmin)/2
		else:
			xmin = (pminx + pmaxx)/2 - (ymax-ymin)/2
			xmax = (pminx + pmaxx)/2 + (ymax-ymin)/2

		#Cut off at border
		xmin = max(0, min(nx, xmin))
		xmax = max(0, min(nx, xmax))
		ymin = max(0, min(nx, ymin))
		ymax = max(0, min(nx, ymax))

		#Move rect
		rect.set_xy((xmin, ymin))
		rect.set_width(xmax-xmin)
		rect.set_height(ymax-ymin)

		#subplot.axis('off')
		#plt.show()
		s1.set_offsets([true_pos[0], true_pos[1]])
		s2.set_offsets([pred_pos[0], pred_pos[1]])
		l[0].set_xdata([pred_pos[0], true_pos[0]]) 
		l[0].set_ydata([pred_pos[1], true_pos[1]])
	
		#Set current axes
		axes.set_xlim([xmin, xmax])
		axes.set_ylim([ymin, ymax])
		axes.invert_yaxis()

		#Text: current error
		#Update text and location 
		txt.set_text = 'Error: %f'%distance_error[idx,c]
		txt.set_position((xmin+0.05*(xmax-xmin), ymin+0.05*(ymax-ymin)))

		#Traces and their correlation over a window
		#Compute window for correlation
		start = max(0, idx - corr_window)
		end = min(nF, start + corr_window)
		tracked_int[1:] = tracked_int[0:-1]
		true_int[1:] = true_int[0:-1]
		tracked_int[0] = grayframe[pred_pos[0], pred_pos[1]]
		true_int[0] = grayframe[true_pos[0], true_pos[1]]
		corr_windowed[c, idx] = np.corrcoef(np.hstack((tracked_int, true_int)).T)

		plt.savefig(fn_out, bbox_inches = 'tight')

avconv = 'avconv -framerate 5 -i ' + imageoutput + 'frame_%03d.png -c:v huffyuv -y'
os.system(avconv + ' ' + imageoutput + 'output.avi')

sample = np.random.choice(range(nC), 150)

#For each cell make plot of correlation and distance over time

#Make plot of distances over time
fn_out = imageoutput + "errors.eps"
plt.clf()
plt.plot(distance_error[:,sample], linewidth = 0.5)
axes = plt.gca()
axes.plot(tracked_thresh*np.ones(nF), linewidth = 3, color = (0, 0, 0), linestyle = 'dashed')
axes.set_ylim([0, 80])
axes.set_ylabel('Tracking error in pixels')
axes.set_xlabel('Frame')

ax2 = axes.twinx()
ax2.plot(trackedprop, linewidth = 2, color = 'k')
ax2.set_ylim([0, 100])
ax2.set_ylabel('Percentage tracked within 6 pixels')
plt.savefig(fn_out, bbox_inches = 'tight')
#plt.show()

