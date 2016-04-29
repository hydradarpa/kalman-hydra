from cuda import * 
from renderer import Renderer
from kalman2 import KalmanFilter 
import dill 
from timeit import timeit 

with open('test/testdata_ones.pkl', 'rb') as f:
	objs = dill.load(f)
(distmesh, vel, flow, nx, im1) = objs

#Introduce some error for testing 
distmesh.p += 2
kf = KalmanFilter(distmesh, im1, flow, cuda=True, vel = vel)
rend = kf.state.renderer
cuda = kf.state.renderer.cudagl 
kf.state.render()

state = kf.state 
y_im = im1 
y_flow = flow 
nx = nx 
deltaX = 3

print 'test_initjacobian_ones'
z_gpu = cuda.initjacobian(im1, flow, test = True)
z_cpu = cuda.initjacobian_CPU(y_im, y_flow, test = True)
print 'CPU:', z_cpu
print 'GPU:', z_gpu
#cuda.total()