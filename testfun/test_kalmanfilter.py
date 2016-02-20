#!/usr/bin/env python
import sys, argparse 
from kalman import KalmanFilter, KFState
from renderer import *
from cuda import *
from distmesh_dyn import DistMesh
import numpy as np 

fn_in ='./video/GCaMP_local_prop.avi'
threshold = 9

capture = VideoStream(fn_in, threshold)
frame = capture.current_frame(backsub = True)
mask, ctrs, fd = capture.backsub()
distmesh = DistMesh(frame)
distmesh.createMesh(ctrs, fd, frame, True)

flowframe = None #capture.backsub(hdf.read())

#Create KalmanFilter object one step at a time (kf = KalmanFilter(distmesh, frame))

#Create KFState object (KFState.__init__)
im = frame 
nx = im.shape[0]
eps_Q = 1
eps_R = 1e-3
_ver = np.array(distmesh.p, np.float32)
_vel = np.ones(_ver.shape, np.float32)

#Create renderer (renderer = Renderer(distmesh, _vel, nx, im))
##############################################################
state = 'texture'
title = 'Hydra tracker. Displaying %s state (space to toggle)' % state
app.Canvas.__init__(self, keys='interactive', title = title)
indices_buffer, outline_buffer, vertex_data = loadMesh(distmesh.p, vel, distmesh.t, nx)
_vbo = gloo.VertexBuffer(vertex_data)

#Setup programs
_program = gloo.Program(VERT_SHADER, FRAG_SHADER)
_program['texture1'] = gloo.Texture2D(im1)
_program.bind(_vbo)
_program_lines = gloo.Program(VERT_SHADER, FRAG_SHADER_LINES)
_program_lines['u_color'] = 1, 1, 0, 1
_program_lines.bind(_vbo)
_program_flowx = gloo.Program(VERT_SHADER, FRAG_SHADER_FLOWX)
_program_flowx['u_color'] = 1, 0, 0, 1
_program_flowx.bind(_vbo)
_program_flowy = gloo.Program(VERT_SHADER, FRAG_SHADER_FLOWY)
_program_flowy['u_color'] = 1, 0, 0, 1
_program_flowy.bind(_vbo)
_program_flow = gloo.Program(VERT_SHADER, FRAG_SHADER_FLOW)
_program_flow['u_color'] = 0, 1, 0, 1
_program_flow.bind(_vbo)

#Create FBOs, attach the color buffer and depth buffer
shape = (nx, nx)
_rendertex1 = gloo.Texture2D((shape + (3,)))
_rendertex2 = gloo.Texture2D((shape + (3,)))
_rendertex3 = gloo.Texture2D((shape + (3,)))
_fbo1 = gloo.FrameBuffer(_rendertex1, gloo.RenderBuffer(shape))
_fbo2 = gloo.FrameBuffer(_rendertex2, gloo.RenderBuffer(shape))
_fbo3 = gloo.FrameBuffer(_rendertex3, gloo.RenderBuffer(shape))

gloo.set_clear_color('black')
_timer = app.Timer('auto', connect=update, start=True)
show()
#CUDAGL.__init__(cudagl = CUDAGL(_rendertex1))

texture = _rendertex1
eps_R = eps_R
width = texture.shape[0]
height = texture.shape[1]
size = texture.shape

import pycuda.gl.autoinit
import pycuda.gl
cuda_gl = pycuda.gl
cuda_driver = pycuda.driver
	
cuda_module = SourceModule("""
__global__ void invert(unsigned char *source, unsigned char *dest)
{
  int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
  int thread_num       = threadIdx.y * blockDim.x + threadIdx.x;
  int threads_in_block = blockDim.x * blockDim.y;
  //Since the image is RGBA we multiply the index 4.
  //We'll only use the first 3 (RGB) channels though
  int idx              = 4 * (threads_in_block * block_num + thread_num);
  dest[idx  ] = 255 - source[idx  ];
  dest[idx+1] = 255 - source[idx+1];
  dest[idx+2] = 255 - source[idx+2];
}
""")
zscore = cuda_module.get_function("invert")

###########################################
kf.compute(capture.gray_frame(), flowframe)
