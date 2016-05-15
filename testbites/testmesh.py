from synthetic import TestMesh
import os.path 
import cv2

flowfields = {}

#Velocity fields:
flowfields['translate_leftup'] = lambda pt,t: (-3, -3)
flowfields['translate_leftup_stretch'] = lambda pt,t: (-3+pt[0]/200., -3+pt[1]/200.)

name = 'square2_gradient'
ff = 'translate_leftup'

fn_in = './synthetictests/' + name + '.png'
v_out = './synthetictests/' + name + '/' + ff
m_out = './synthetictests/' + name + '/' + ff + '_mesh.txt'

if os.path.isfile(fn_in):
	if not os.path.isdir('./synthetictests/' + name):
		os.makedirs('./synthetictests/' + name)
	#Load image as grayscale
	img = cv2.imread(fn_in,0)
	tm = TestMesh(img, flowfields[ff], gridsize = 50)
	tm.run(v_out, m_out)
else:
	print "Cannot find", fn_in, ", exiting."