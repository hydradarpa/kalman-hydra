#################################
#measure image similarity on GPU#
#################################

#lansdell. Feb 15th 2016

import numpy as np, Image
import sys, time, os

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
from OpenGL.GL.ARB.pixel_buffer_object import *

import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl
import pycuda
from pycuda.compiler import SourceModule

from vispy import gloo 

import pdb 
import cv2
from matplotlib import pyplot as plt

class CUDAGL:
	def __init__(self, texture, fbo):
		self._fbo1 = fbo 
		self.texture = texture
		self.width = texture.shape[0]
		self.height = texture.shape[1]
		self.size = texture.shape

		import pycuda.gl.autoinit
		import pycuda.gl
		cuda_gl = pycuda.gl
		cuda_driver = pycuda.driver
	
		cuda_module = SourceModule("""
		__global__ void zscore(unsigned char *y_tilde, unsigned char *yp_tilde, unsigned char *z)
		{
		  int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
		  int thread_num       = threadIdx.y * blockDim.x + threadIdx.x;
		  int threads_in_block = blockDim.x * blockDim.y;
		  //Since the image is RGBA we multiply the index 4.
		  //We'll only use the first 3 (RGB) channels though
		  int idx              = 4 * (threads_in_block * block_num + thread_num);
		  //y_tilde[idx  ] = 255 - y_im[idx  ];
		  //y_tilde[idx+1] = 255 - y_im[idx+1];
		  //y_tilde[idx+2] = 255 - y_im[idx+2];
		}
		""")
		self.zscore = cuda_module.get_function("zscore")
		# The argument "PP" indicates that the zscore function will take two PBOs as arguments
		self.zscore.prepare("PPP")
		# create y_tilde and y_im pixel buffer objects for processing
		self._createPBOs()

	def _createPBOs(self):
		global y_tilde_pbo, yp_tilde_pbo, z_tilde_pbo, pycuda_y_tilde_pbo, pycuda_yp_tilde_pbo, pycuda_z_tilde_pbo
		num_texels = self.width*self.height
		data = np.zeros((num_texels,4),np.uint8)
		y_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, y_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_y_tilde_pbo = cuda_gl.BufferObject(long(y_tilde_pbo))

		yp_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, yp_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_yp_tilde_pbo = cuda_gl.BufferObject(long(yp_tilde_pbo))

		z_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, z_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_z_tilde_pbo = cuda_gl.BufferObject(long(z_tilde_pbo))

	def _destroy_PBOs():
		global y_tilde_pbo, yp_tilde_pbo, z_tilde_pbo, pycuda_y_tilde_pbo, pycuda_yp_tilde_pbo, pycuda_z_tilde_pbo
		for pbo in [y_tilde_pbo, yp_tilde_pbo, z_tilde_pbo]:
			glBindBuffer(GL_ARRAY_BUFFER, long(pbo))
			glDeleteBuffers(1, long(pbo));
			glBindBuffer(GL_ARRAY_BUFFER, 0)
		y_tilde_pbo,yp_tilde_pbo,z_tilde_pbo,pycuda_y_tilde_pbo,pycuda_yp_tilde_pbo,pycuda_z_tilde_pbo = [None]*6    

	def jz(self, y_im):
		global pycuda_y_tilde_pbo,y_tilde_pbo, pycuda_yp_tilde_pbo, yp_tilde_pbo, z_tilde_pbo, pycuda_z_tilde_pbo
		assert yp_tilde_pbo is not None
		# tell cuda we are going to get into these buffers
		pycuda_yp_tilde_pbo.unregister()
		#Load yp_tilde texture info
		#activate y_tilde buffer
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(yp_tilde_pbo))
		#Needed? is according to http://stackoverflow.com/questions/10507215/how-to-copy-a-texture-into-a-pbo-in-pyopengl
		#glBufferData(GL_PIXEL_PACK_BUFFER_ARB,
		#         bytesize,
		#         None, self.usage)
		#read data into pbo. note: use BGRA format for optimal performance
		glEnable(GL_TEXTURE_2D)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, self.texture.id)
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
		#    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
		#         w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)
		pycuda_yp_tilde_pbo = cuda_gl.BufferObject(long(yp_tilde_pbo))	
		#run the Cuda kernel
		z = self._process()
		return z

	def _process(self):
		""" Use PyCuda """
		grid_dimensions = (self.width//10,self.height//10)
		y_tilde_mapping = pycuda_y_tilde_pbo.map()
		yp_tilde_mapping = pycuda_yp_tilde_pbo.map()
		z_tilde_mapping = pycuda_z_tilde_pbo.map()
		self.zscore.prepared_call(grid_dimensions, (10, 10, 1),
				y_tilde_mapping.device_ptr(),
				yp_tilde_mapping.device_ptr(),z_tilde_mapping.device_ptr())
		cuda_driver.Context.synchronize()
		y_tilde_mapping.unmap()
		yp_tilde_mapping.unmap()
		z_tilde_mapping.unmap()
		#Get result from CUDA...
		return 0

	def initjacobian(self, y_im, y_flow):
		#Compute z = y_im - y_tilde 
		#Copy z to GPU 
		#Copy y_tilde to GPU 

		# tell cuda we are going to get into these buffers
		pycuda_y_tilde_pbo.unregister()
		pycuda_z_tilde_pbo.unregister()

		#Load y_tilde texture info
		#activate y_tilde buffer
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_tilde_pbo))
		#Needed? is according to http://stackoverflow.com/questions/10507215/how-to-copy-a-texture-into-a-pbo-in-pyopengl
		#glBufferData(GL_PIXEL_PACK_BUFFER_ARB,
		#         bytesize,
		#         None, self.usage)
		#read data into pbo. note: use BGRA format for optimal performance
		glEnable(GL_TEXTURE_2D)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, self.texture.id)
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		#Load y_im (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_im_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height, y_im, GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		pycuda_y_tilde_pbo = cuda_gl.BufferObject(long(y_tilde_pbo))
		pycuda_z_tilde_pbo = cuda_gl.BufferObject(long(z_tilde_pbo))
		return (0,0)