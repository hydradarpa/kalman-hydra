from cuda import * 
from renderer import Renderer
from kalman2 import KalmanFilter 
import dill 

with open('test/testdata_ones.pkl', 'rb') as f:
	objs = dill.load(f)
(distmesh, vel, flow, nx, im1) = objs
kf = KalmanFilter(distmesh, im1, flow, cuda=True, vel = vel)
rend = kf.state.renderer
cuda = kf.state.renderer.cudagl 
kf.state.render()

state = kf.state 
y_im = im1 
y_flow = flow 
nx = nx 
deltaX = 3

eps = 1e-6
print 'test_initjacobian_ones'
#cuda.initjacobian(im1, flow)

cuda.total()