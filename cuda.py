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
	def __init__(self, texture, eps_R, fbo):
		self._fbo1 = fbo 
		self.texture = texture
		self.eps_R = eps_R
		self.width = texture.shape[0]
		self.height = texture.shape[1]
		self.size = texture.shape

		import pycuda.gl.autoinit
		import pycuda.gl
		cuda_gl = pycuda.gl
		cuda_driver = pycuda.driver
	
		cuda_module = SourceModule("""
		__global__ void zscore(unsigned char *y_tilde, unsigned char *y_im)
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
		# The argument "PP" indicates that the invert function will take two PBOs as arguments
		self.zscore.prepare("PP")
		# create y_tilde and y_im pixel buffer objects for processing
		self._createPBOs()

	def _createPBOs(self):
		global y_tilde_pbo, y_im_pbo, pycuda_y_tilde_pbo, pycuda_y_im_pbo
		num_texels = self.width*self.height
		data = np.zeros((num_texels,4),np.uint8)
		y_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, y_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_y_tilde_pbo = cuda_gl.BufferObject(long(y_tilde_pbo))

		y_im_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, y_im_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_y_im_pbo = cuda_gl.BufferObject(long(y_im_pbo))
	
	def _destroy_PBOs():
		global y_tilde_pbo, y_im_pbo, pycuda_y_tilde_pbo, pycuda_y_im_pbo
		for pbo in [y_tilde_pbo, y_im_pbo]:
			glBindBuffer(GL_ARRAY_BUFFER, long(pbo))
			glDeleteBuffers(1, long(pbo));
			glBindBuffer(GL_ARRAY_BUFFER, 0)
		y_tilde_pbo,y_im_pbo,pycuda_y_tilde_pbo,pycuda_y_im_pbo = [None]*4    

	def z(self, y_im):
		global pycuda_y_tilde_pbo,y_tilde_pbo, y_im_pbo,pycuda_y_im_pbo
		assert y_tilde_pbo is not None

		# tell cuda we are going to get into these buffers
		pycuda_y_tilde_pbo.unregister()
		pycuda_y_im_pbo.unregister()
		#Load y_tilde texture info
		#activate y_tilde buffer
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_tilde_pbo))
		#Needed? (is according to http://stackoverflow.com/questions/10507215/how-to-copy-a-texture-into-a-pbo-in-pyopengl)
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
	
		#pbo_buffer = glGenBuffers(1) # generate 1 buffer reference
		#glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer) # binding to this buffer
		#glBufferData(GL_PIXEL_UNPACK_BUFFER, imWidth*imHeight, pixels, GL_STREAM_DRAW) # Allocate the buffer
		#bsize = glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE) # Check allocated buffer size
		#assert(bsize == imWidth*imHeight)
		#glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0) # Unbind

		#Load y_im (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_im_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height, y_im, GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		pycuda_y_tilde_pbo = cuda_gl.BufferObject(long(y_tilde_pbo))
		pycuda_y_im_pbo = cuda_gl.BufferObject(long(y_im_pbo))
	
		#run the Cuda kernel
		z = self._process()
		return z

	def _process(self):
		""" Use PyCuda """
		grid_dimensions = (self.width//10,self.height//10)
		y_tilde_mapping = pycuda_y_tilde_pbo.map()
		y_im_mapping = pycuda_y_im_pbo.map()
		self.zscore.prepared_call(grid_dimensions, (10, 10, 1),
				y_tilde_mapping.device_ptr(),
				y_im_mapping.device_ptr())
		cuda_driver.Context.synchronize()
		y_tilde_mapping.unmap()
		y_im_mapping.unmap()

		#Get result from CUDA...
		return 0

	#Test code on the CPU
	def z_CPU(self, y_im):
		#Copy y_tilde pixels to CPU memory
		with self._fbo1:
			y_tilde = gloo.read_pixels()
			#Not sure why I can't get this one to work...
			#y_tilde = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)
			#y_tilde = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
			#y_tilde = np.fromstring(y_tilde, dtype=np.uint8).reshape((self.width, self.height, 3))
		#print y_tilde.shape 
		#plt.imshow(y_im-y_tilde[:,:,0]>0),plt.colorbar(),plt.show()
		#cv2.imshow('',y_im)
		#k = cv2.waitKey(30) & 0xff
		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")
		#so much simpler...
		if len(y_im.shape) == 3:
			diff = y_tilde[:,:,0]-y_im[:,:,0]
		else:
			diff = y_tilde[:,:,0]-y_im

		diff = diff# + np.random.normal(size = diff.shape)*self.eps_R
		z = np.sum(np.multiply(diff, diff)/256./256)#./self.eps_R/self.eps_R)
		#print z 
		#input()
		return z