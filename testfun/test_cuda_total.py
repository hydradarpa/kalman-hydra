from cuda import * 
from renderer import Renderer
from kalman2 import KalmanFilter 
import dill 
from timeit import timeit 

with open('test/testdata_ones.pkl', 'rb') as f:
	objs = dill.load(f)
(distmesh, vel, flow, nx, im1) = objs

#Introduce some error for testing 
#distmesh.p += 2
kf = KalmanFilter(distmesh, im1, flow, cuda=True, vel = vel)
rend = kf.state.renderer
cuda = kf.state.renderer.cudagl 
kf.state.render()

state = kf.state 
y_im = im1 
y_flow = flow 
nx = nx 
deltaX = -2

z_gpu = cuda.initjacobian(im1, flow, test = True)
z_cpu = cuda.initjacobian_CPU(y_im, y_flow, test = True)
print 'Test initjacobian'
print 'CPU:', z_cpu
print 'GPU:', z_gpu

for idx in range(16):
	#Perturb vertices a bit and rerender
	#idx = 10
	kf.state.X[idx,0] += deltaX
	kf.state.refresh()
	kf.state.render()
	kf.state.X[idx,0] -= deltaX
	
	jz_gpu = cuda.jz()
	jz_cpu = cuda.jz_CPU()
	print 'Test jz'
	print 'CPU:', jz_cpu
	print 'GPU:', jz_gpu
	
deltaX = 100
#i = 1; j = 14
for i in range(16):
	for j in range(16):
		j_gpu = cuda.j(kf.state, deltaX, i, j)
		j_cpu = cuda.j_CPU(kf.state, deltaX, i, j)
		print 'Test j'
		print 'CPU:', j_cpu
		print 'GPU:', j_gpu
