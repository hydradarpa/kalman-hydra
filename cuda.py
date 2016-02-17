#################################
#measure image similarity on GPU#
#################################

#lansdell. Feb 15th 2016

import numpy as np, Image
import sys, time, os
import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl
import pycuda
from pycuda.compiler import SourceModule

class CUDAGL:
	def __init__(self, texture):

		self.imWidth = pixels.shape[0]
		self.imHeight = pixels.shape[1]
		if pycuda.VERSION[0] >= 2011:
			self.ver2011 = True
		#Format pixel data as array... is this needed?
		self.pixels = pixels
		self.array = cuda_driver.matrix_to_array(pixels, "C") # C-style instead of Fortran-style: row-major
		self.pixels.fill(0) # Resetting the array to 0	
		#Create PBO
		self.pbo_buffer = glGenBuffers(1) # generate 1 buffer reference
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_buffer) # binding to this buffer

		#Init taken from
		#http://stackoverflow.com/questions/10507215/how-to-copy-a-texture-into-a-pbo-in-pyopengl
		self.usage = usage
		if isinstance(size, tuple):
			size = size[0] * size[1] * self.imageformat.planecount
		bytesize = self.imageformat.get_bytesize_per_plane() * size
		glBindBuffer(self.arraytype, self.id)
		glBufferData(self.arraytype, bytesize, None, self.usage)
		glBindBuffer(self.arraytype, 0)

		#Unpack pixels to PBO
		glBufferData(GL_PIXEL_UNPACK_BUFFER, self.imWidth*self.imHeight, self.pixels, GL_STREAM_DRAW) # Allocate the buffer
		bsize = glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE) # Check allocated buffer size
		assert(bsize == self.imWidth*self.imHeight)
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0) # Unbind
		#Register PBO with CUDA
		if ver2011:
			self.cuda_pbo_resource = pycuda.gl.RegisteredBuffer(int(self.pbo_buffer), cuda_gl.graphics_map_flags.WRITE_DISCARD)
		else:
			self.cuda_pbo_resource = cuda_gl.BufferObject(int(self.pbo_buffer)) # Mapping GLBuffer to cuda_resource
		
		#Not sure what this does...
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1) # 1-byte row alignment
		glPixelStorei(GL_PACK_ALIGNMENT, 1) # 1-byte row alignment        

	def z(self, y_texture, y_im, eps_R):
		#Unpack texture to PBO
		bytesize = (self.size[0] * self.size[1] *
					self.imageformat.planecount *
					self.imageformat.get_bytesize_per_plane())
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, self.id)
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB,
					bytesize,
					None, self.usage)
		glEnable(texture.target)
		glActiveTexture(GL_TEXTURE0_ARB)
		glBindTexture(texture.target, texture.id)
		OpenGL.raw.GL.glGetTexImage(texture.target, 0, texture.gl_imageformat,
									texture.gl_dtype, ctypes.c_void_p(0))
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(texture.target)


		#Map PBO to CUDA

		#Run CUDA zscore
		#CUDA: 
		#Subtract one from the other
		#Add noise of std eps_R to difference
		#Compute ssr of two images
		#normalize by variance (eps_R^2)

		#Unmap CUDA PBO

		return 0 

	def sobelFilter(odata, iw, ih):
		# Texture and shared memory with fixed BlockSize
		sm = SourceModule("""
			texture<unsigned char, 2> tex;
			extern __shared__ unsigned char LocalBlock[];
			#define RADIUS 1
			#define BlockWidth 80
			#define SharedPitch 384
			__device__ unsigned char
			ComputeSobel(unsigned char ul, // upper left
						 unsigned char um, // upper middle
						 unsigned char ur, // upper right
						 unsigned char ml, // middle left
						 unsigned char mm, // middle (unused)
						 unsigned char mr, // middle right
						 unsigned char ll, // lower left
						 unsigned char lm, // lower middle
						 unsigned char lr, // lower right
						 float fScale )
			{
				short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
				short Vert = ul + 2*um + ur - ll - 2*lm - lr;
				short Sum = (short) (fScale*(::abs(int(Horz))+::abs(int(Vert))));
				if ( Sum < 0 ) return 0; else if ( Sum > 0xff ) return 0xff;
				return (unsigned char) Sum;
			}
			__global__ void
			SobelShared( int* pSobelOriginal, unsigned short SobelPitch,
						 short w, short h, float fScale )
			{
				short u = 4*blockIdx.x*BlockWidth;
				short v = blockIdx.y*blockDim.y + threadIdx.y;
				short ib;
				int SharedIdx = threadIdx.y * SharedPitch;
				for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {
					LocalBlock[SharedIdx+4*ib+0] = tex2D( tex,
						(float) (u+4*ib-RADIUS+0), (float) (v-RADIUS) );
					LocalBlock[SharedIdx+4*ib+1] = tex2D( tex,
						(float) (u+4*ib-RADIUS+1), (float) (v-RADIUS) );
					LocalBlock[SharedIdx+4*ib+2] = tex2D( tex,
						(float) (u+4*ib-RADIUS+2), (float) (v-RADIUS) );
					LocalBlock[SharedIdx+4*ib+3] = tex2D( tex,
						(float) (u+4*ib-RADIUS+3), (float) (v-RADIUS) );
				}
				if ( threadIdx.y < RADIUS*2 ) {
					//
					// copy trailing RADIUS*2 rows of pixels into shared
					//
					SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;
					for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {
						LocalBlock[SharedIdx+4*ib+0] = tex2D( tex,
							(float) (u+4*ib-RADIUS+0), (float) (v+blockDim.y-RADIUS) );
						LocalBlock[SharedIdx+4*ib+1] = tex2D( tex,
							(float) (u+4*ib-RADIUS+1), (float) (v+blockDim.y-RADIUS) );
						LocalBlock[SharedIdx+4*ib+2] = tex2D( tex,
							(float) (u+4*ib-RADIUS+2), (float) (v+blockDim.y-RADIUS) );
						LocalBlock[SharedIdx+4*ib+3] = tex2D( tex,
							(float) (u+4*ib-RADIUS+3), (float) (v+blockDim.y-RADIUS) );
					}
				}
				__syncthreads();
				u >>= 2;    // index as uchar4 from here
				uchar4 *pSobel = (uchar4 *) (((char *) pSobelOriginal)+v*SobelPitch);
				SharedIdx = threadIdx.y * SharedPitch;
				for ( ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x ) {
					unsigned char pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
					unsigned char pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+1];
					unsigned char pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+2];
					unsigned char pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+0];
					unsigned char pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+1];
					unsigned char pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+2];
					unsigned char pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+0];
					unsigned char pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+1];
					unsigned char pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+2];
					uchar4 out;
					out.x = ComputeSobel(pix00, pix01, pix02,
										 pix10, pix11, pix12,
										 pix20, pix21, pix22, fScale );
					pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
					pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
					pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
					out.y = ComputeSobel(pix01, pix02, pix00,
										 pix11, pix12, pix10,
										 pix21, pix22, pix20, fScale );
					pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
					pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
					pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
					out.z = ComputeSobel( pix02, pix00, pix01,
										  pix12, pix10, pix11,
										  pix22, pix20, pix21, fScale );
					pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
					pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
					pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
					out.w = ComputeSobel( pix00, pix01, pix02,
										  pix10, pix11, pix12,
										  pix20, pix21, pix22, fScale );
					if ( u+ib < w/4 && v < h ) {
						pSobel[u+ib] = out;
					}
				}
				__syncthreads();
			}
		""")
		cuda_function = sm.get_function("SobelShared")
	
		texref = sm.get_texref("tex")
		texref.set_array(array)
		texref.set_flags(cuda_driver.TRSA_OVERRIDE_FORMAT)
	
		# fixed BlockSize Launch
		RADIUS = 1
		threads = (16, 4, 1)
		BlockWidth = 80 # Do not change!
		blocks = (iw/(4*BlockWidth)+(0!=iw%(4*BlockWidth)),
							   ih/threads[1]+(0!=ih%threads[1]) )
		SharedPitch = ~0x3f & (4*(BlockWidth+2*RADIUS)+0x3f);
		sharedMem = SharedPitch*(threads[1]+2*RADIUS);
		iw = iw & ~3
		cuda_function(np.intp(odata), np.uint16(iw), np.int16(iw), np.int16(ih), np.float32(scale), texrefs=[texref],block=threads, grid=blocks, shared=sharedMem)