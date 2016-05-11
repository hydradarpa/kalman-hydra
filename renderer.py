#################################
#gloo renderer for Kalman filter#
#################################

#lansdell. Feb 11th 2016

import numpy as np 
from vispy import gloo
from vispy import app
from imgproc import findObjectThreshold
import cv2 
from time import gmtime, strftime

from cuda import CUDAGL 

CV_MAT_DEPTH_MASK = 7
CV_CN_SHIFT = 3

VERT_SHADER = """ // simple vertex shader

attribute vec3 a_position;
attribute vec3 a_velocity;
attribute vec2 a_texcoord;
uniform vec4 u_color;
varying vec4 v_color;
varying vec3 v_vel;

void main (void) {
	v_vel = a_velocity;
	v_color = u_color;
	// Pass tex coords
	gl_TexCoord[0] = vec4(a_texcoord.x, a_texcoord.y, 0.0, 0.0);
	// Calculate position
	gl_Position = vec4(a_position.x, a_position.y, a_position.z, 1.0);
}
"""

FRAG_SHADER = """ // simple fragment shader
uniform sampler2D texture1;

void main()
{
	gl_FragColor = texture2D(texture1, gl_TexCoord[0].st);
}
"""

FRAG_SHADER_RED = """ // simple fragment shader
uniform sampler2D texture1;

void main()
{
	vec4 tex1 = texture2D(texture1, gl_TexCoord[0].st);
	gl_FragColor = vec4(tex1.r, 0, 0, tex1.a);
}
"""

FRAG_SHADER_GREEN = """ // simple fragment shader
uniform sampler2D texture1;

void main()
{
	vec4 tex1 = texture2D(texture1, gl_TexCoord[0].st);
	gl_FragColor = vec4(0, tex1.g, 0, tex1.a);
}
"""

FRAG_SHADER_FLOWX = """ // simple fragment shader
varying vec3 v_vel;
varying vec4 v_color;
float c_zero = 0.0;

void main()
{
	gl_FragColor = vec4(c_zero, c_zero, c_zero, 1.0);
	gl_FragColor.r = v_vel.x;

	//vec3 pix;
	//pix.r = v_vel.x;//+5)/10;
	//pix.g = v_vel.y;//+5)/10;
	//pix.b = 0.0;
	//gl_FragColor.rgb = pix;
	//gl_FragColor.a = 1.0;
}
"""


FRAG_SHADER_FLOWY = """ // simple fragment shader
varying vec3 v_vel;
varying vec4 v_color;
const float c_zero = 0.0;

void main()
{
	gl_FragColor = vec4(c_zero, c_zero, c_zero, 1.0);
	gl_FragColor.r = v_vel.y;
}
"""

FRAG_SHADER_FLOW = """ // fragment shader to convert flow to hue/saturation
varying vec3 v_vel;
varying vec4 v_color;

// relative lengths of color transitions:
// these are chosen based on perceptual similarity
// (e.g. one can distinguish more shades between red and yellow
//  than between yellow and green)
const int RY = 15;
const int YG = 6;
const int GC = 4;
const int CB = 11;
const int BM = 13;
const int MR = 6;
const int NCOLS = RY + YG + GC + CB + BM + MR;
vec3 colorWheel[NCOLS];

const float maxrad = 5;
const float PI = 3.14159265;

void main() {
	float fx = v_vel.x;
	float fy = v_vel.y;
	int k = 0;
	for (int i = 0; i < RY; ++i, ++k)
		colorWheel[k] = vec3(255, 255 * i / RY, 0);
	for (int i = 0; i < YG; ++i, ++k)
		colorWheel[k] = vec3(255 - 255 * i / YG, 255, 0);
	for (int i = 0; i < GC; ++i, ++k)
		colorWheel[k] = vec3(0, 255, 255 * i / GC);
	for (int i = 0; i < CB; ++i, ++k)
		colorWheel[k] = vec3(0, 255 - 255 * i / CB, 255);
	for (int i = 0; i < BM; ++i, ++k)
		colorWheel[k] = vec3(255 * i / BM, 0, 255);
	for (int i = 0; i < MR; ++i, ++k)
		colorWheel[k] = vec3(255, 0, 255 - 255 * i / MR);
	float rad = sqrt(fx * fx + fy * fy);
	float a = atan(-fy, -fx) / PI;
	float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
	int k0 = int(fk);
	int k1 = int(mod(k0 + 1, NCOLS));
	float f = fk - k0;
	vec3 pix;
	for (int b = 0; b < 3; b++)
	{
		float col0 = colorWheel[k0][b] / 255.0f;
		float col1 = colorWheel[k1][b] / 255.0f;
		float col = (1 - f) * col0 + f * col1;
		if (rad <= maxrad)
			col = rad * col / maxrad; // increase saturation with radius
		//else
		//	col *= .75; // out of range
		pix[2 - b] = col;
	}

	gl_FragColor.rgb = pix;
	gl_FragColor.a = 1.0;

	/* //Simple shader
	vec3 pix;
	pix.r = (fx+5)/10;
	pix.g = (fy+5)/10;
	pix.b = 0.0;
	gl_FragColor.rgb = pix;
	gl_FragColor.a = 1.0;
	*/
}
"""

FRAG_SHADER_LINES = """ // simple fragment shader
varying vec4 v_color;

void main()
{
	gl_FragColor = v_color;
}
"""

class Renderer(app.Canvas):

	def __init__(self, distmesh, vel, flow, nx, im1, cuda, showtracking = False):
		self.cuda = cuda
		self.showtracking = showtracking 
		self.state = 'texture'
		title = 'Hydra tracker. Displaying %s state (space to toggle)' % self.state
		app.Canvas.__init__(self, keys='interactive', title = title, show = showtracking)
		self.size = (nx, nx)
		self.indices_buffer, self.outline_buffer, self.vertex_data, self.quad_data, self.quad_buffer = self.loadMesh(distmesh.p, vel, distmesh.t, nx)
		self._vbo = gloo.VertexBuffer(self.vertex_data)
		self._quad = gloo.VertexBuffer(self.quad_data)
		self.current_frame = im1
		self.current_texture = gloo.Texture2D(im1)
		self.init_texture = gloo.Texture2D(im1)

		self.current_flowx = flow[:,:,0]
		self.current_flowy = flow[:,:,1]
		self.current_fx_texture = gloo.Texture2D(flow[:,:,0], format="luminance", internalformat="r32f")
		self.current_fy_texture = gloo.Texture2D(flow[:,:,1], format="luminance", internalformat="r32f")

		#Setup programs
		self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
		self._program['texture1'] = self.init_texture
		self._program.bind(self._vbo)
		self._program_lines = gloo.Program(VERT_SHADER, FRAG_SHADER_LINES)
		self._program_lines['u_color'] = 0, 1, 1, 1
		self._program_lines.bind(self._vbo)
		self._program_flowx = gloo.Program(VERT_SHADER, FRAG_SHADER_FLOWX)
		self._program_flowx['u_color'] = 1, 0, 0, 1
		self._program_flowx.bind(self._vbo)
		self._program_flowy = gloo.Program(VERT_SHADER, FRAG_SHADER_FLOWY)
		self._program_flowy['u_color'] = 1, 0, 0, 1
		self._program_flowy.bind(self._vbo)
		self._program_flow = gloo.Program(VERT_SHADER, FRAG_SHADER_FLOW)
		self._program_flow['u_color'] = 0, 1, 0, 1
		self._program_flow.bind(self._vbo)
		self._program_red = gloo.Program(VERT_SHADER, FRAG_SHADER_RED)
		self._program_green = gloo.Program(VERT_SHADER, FRAG_SHADER_GREEN)

		#Create FBOs, attach the color buffer and depth buffer
		self.shape = (nx, nx)
		self._rendertex1 = gloo.Texture2D((self.shape + (1,)), format="luminance", internalformat="r8")
		self._rendertex2 = gloo.Texture2D((self.shape + (1,)), format="luminance", internalformat="r32f")
		self._rendertex3 = gloo.Texture2D((self.shape + (1,)), format="luminance", internalformat="r32f")
		self._fbo1 = gloo.FrameBuffer(self._rendertex1, gloo.RenderBuffer(self.shape))
		self._fbo2 = gloo.FrameBuffer(self._rendertex2, gloo.RenderBuffer(self.shape))
		self._fbo3 = gloo.FrameBuffer(self._rendertex3, gloo.RenderBuffer(self.shape))

		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")
		gloo.set_viewport(0, 0, nx, nx)
		gloo.set_clear_color('black')
		self._timer = app.Timer('auto', connect=self.update, start=True)
		if showtracking:
			self.show()
		self.on_draw(None)
		#print self._rendertex1.id
		#print self.context.shared._parser._objects
		a=self.context.shared._parser.get_object(self._rendertex1.id)._handle
		b=self.context.shared._parser.get_object(self._rendertex2.id)._handle
		c=self.context.shared._parser.get_object(self._rendertex3.id)._handle
		self.cudagl = CUDAGL(self._rendertex1, self._rendertex2, self._rendertex3, self._fbo1, self._fbo2, self._fbo3, a, b, c, cuda)

	def on_resize(self, event):
		width, height = event.physical_size
		#gloo.set_viewport(0, 0, width, height)

	def on_draw(self, event):
		#Render the current positions to FBO1 
		with self._fbo1:
			gloo.clear()
			self._program.draw('triangles', self.indices_buffer)
		#Render the current velocities to FBO2
		with self._fbo2:
			gloo.clear()
			self._program_flowx.draw('triangles', self.indices_buffer)
		with self._fbo3:
			gloo.clear()
			self._program_flowy.draw('triangles', self.indices_buffer)

		#Turn on additive blending
		gloo.set_state('additive')
		gloo.clear()
		#Summary render to main screen
		if self.state == 'texture' or self.state == 'raw':
			self._program.draw('triangles', self.indices_buffer)
		elif self.state == 'flow':
			self._program_flow.draw('triangles', self.indices_buffer)			
		else:
			self._program_red['texture1'] = self.current_texture
			self._program_red.bind(self._quad)
			self._program_red.draw('triangles', self.quad_buffer)
			self._program_green.bind(self._vbo)
			self._program_green['texture1'] = self.init_texture
			self._program_green.draw('triangles', self.indices_buffer)
		#Draw wireframe, too
		if self.state != 'raw':
			self._program_lines.draw('lines', self.outline_buffer)

	def on_key_press(self, event):
		if event.key in [' ']:
			if self.state == 'flow':
				self.state = 'raw'
			elif self.state == 'raw':
				self.state = 'overlay'
			elif self.state == 'overlay':
				self.state = 'texture'
			else:
				self.state = 'flow'
			self.title = 'Hydra tracker. Displaying %s state (space to toggle)' % self.state
			self.update()
		if event.key in ['s']:
			self.screenshot()

	def screenshot(self, saveall = False, basename = 'screenshot'):
		with self._fbo2:
			pixels = gloo.read_pixels(out_type = np.float32)[:,:,0]
			fn = './' + basename + '_flowx_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.png'
			print np.max(pixels)
			cv2.imwrite(fn, (255.*(pixels-np.min(pixels))/(np.max(pixels)-np.min(pixels))).astype(int))
		with self._fbo3:
			pixels = gloo.read_pixels(out_type = np.float32)[:,:,0]
			fn = './' + basename + '_flowy_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.png'
			cv2.imwrite(fn, (255.*(pixels-np.min(pixels))/(np.max(pixels)-np.min(pixels))).astype(int))
		if not saveall:
			pixels = gloo.read_pixels()
			fn = './' + basename + '_' + self.state + '_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.png'
			print 'Saving screenshot to ' + fn
			cv2.imwrite(fn, pixels)
		else:
			oldstate = self.state
			#change render mode, rerender, and save
			for state in ['flow', 'raw', 'overlay', 'texture']:
				self.state = state
				self.on_draw(None)
				pixels = gloo.read_pixels()
				fn = './' + basename + '_' + state + '_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.png'
				#print 'Saving screenshot to ' + fn
				cv2.imwrite(fn, pixels)
			self.state = oldstate
			self.update()

	def error(self, state, y_im, y_flow):
		state.refresh()
		state.render()
		with self._fbo1:
			pixels = gloo.read_pixels()
		with self._fbo2:
			fx = gloo.read_pixels(out_type = np.float32)
		with self._fbo3:
			fy = gloo.read_pixels(out_type = np.float32)

		e_im = np.sum(np.multiply(y_im-pixels[:,:,0], y_im-pixels[:,:,0]))
		e_fx = np.sum(np.multiply(y_flow[:,:,0]-fx[:,:,0], y_flow[:,:,0]-fx[:,:,0]))
		e_fy = np.sum(np.multiply(y_flow[:,:,1]+fy[:,:,0], y_flow[:,:,1]+fy[:,:,0]))
		return e_im, e_fx, e_fy, fx, fy

	def update_vertex_buffer(self, vertices, velocities):
		verdata = np.zeros((self.nP,3))
		veldata = np.zeros((self.nP,3))
		#rescale
		verdata[:,0:2] = 2*vertices/self.nx-1
		verdata[:,1] = -verdata[:,1]
		#veldata[:,0:2] = 2*velocities/self.nx
		veldata[:,0:2] = velocities
		veldata[:,1] = -veldata[:,1]
		self.vertex_data['a_position'] = verdata
		self.vertex_data['a_velocity'] = veldata 
		self._vbo = gloo.VertexBuffer(self.vertex_data)
		self._program.bind(self._vbo)
		self._program_lines.bind(self._vbo)
		self._program_flowx.bind(self._vbo)
		self._program_flowy.bind(self._vbo)
		self._program_flow.bind(self._vbo)

	#Load mesh data
	def loadMesh(self, vertices, velocities, triangles, nx):
		# Create vetices and texture coords, combined in one array for high performance
		self.nP = vertices.shape[0]
		self.nT = triangles.shape[0]
		self.nx = nx
		vertex_data = np.zeros(self.nP, dtype=[('a_position', np.float32, 3),
			('a_texcoord', np.float32, 2), ('a_velocity', np.float32, 3)])
		verdata = np.zeros((self.nP,3))
		veldata = np.zeros((self.nP,3))
		uvdata = np.zeros((self.nP,2))
		outlinedata = np.zeros((self.nT, 6))
		#rescale
		verdata[:,0:2] = 2*vertices/nx-1
		verdata[:,1] = -verdata[:,1]
		#veldata[:,0:2] = 2*velocities/nx-1
		veldata[:,0:2] = velocities
		veldata[:,1] = -veldata[:,1]
		uvdata = vertices/nx
		vertex_data['a_position'] = verdata
		vertex_data['a_texcoord'] = uvdata 
		vertex_data['a_velocity'] = veldata 
		indices = triangles.reshape((1,-1)).astype(np.uint16)
		indices_buffer = gloo.IndexBuffer(indices)
	
		for idx, t in enumerate(triangles):
			outlinedata[idx,0] = t[0]
			outlinedata[idx,1] = t[1]
			outlinedata[idx,2] = t[1]
			outlinedata[idx,3] = t[2]
			outlinedata[idx,4] = t[2]
			outlinedata[idx,5] = t[0]
		outline = outlinedata.reshape((1,-1)).astype(np.uint16)
		outline_buffer = gloo.IndexBuffer(outline)

		quad_data = np.zeros(4, dtype=[('a_position', np.float32, 3),
				('a_texcoord', np.float32, 2), ('a_velocity', np.float32, 3)])
		quad_ver = np.array([[-1.0, 1.0, 0.0],  [-1.0, -1.0, 0.0],
							 [ 1.0, 1.0, 0.0], [ 1.0, -1.0, 0.0,] ], np.float32)
		quad_coord = np.array([  [0.0, 0.0], [0.0, 1.0],
								[1.0, 0.0], [1.0, 1.0] ], np.float32)
		quad_vel = np.zeros((4,3))
		quad_data['a_texcoord'] = quad_coord 
		quad_data['a_position'] = quad_ver 
		quad_data['a_velocity'] = quad_vel 
		quad_triangles = np.array([[0, 1, 2], [1, 2, 3]])
		quad_indices = quad_triangles.reshape((1,-1)).astype(np.uint16)
		quad_buffer = gloo.IndexBuffer(quad_indices)

		return indices_buffer, outline_buffer, vertex_data, quad_data, quad_buffer

	def update_frame(self, y_im, y_flow):
		self.current_frame = y_im 
		self.current_texture = gloo.Texture2D(y_im)
		self.current_flowx = y_flow[:,:,0] 
		self.current_fx_texture = gloo.Texture2D(y_flow[:,:,0], format="luminance", internalformat="r32f")
		self.current_flowy = y_flow[:,:,1] 
		self.current_fy_texture = gloo.Texture2D(y_flow[:,:,1], format="luminance", internalformat="r32f")

	def get_flow(self):
		self.on_draw(None)
		with self._fbo2:
			flowx = gloo.read_pixels(out_type = np.float32)
		with self._fbo3:
			flowy = gloo.read_pixels(out_type = np.float32)
		return (flowx, flowy)

	def initjacobian(self, y_im, y_flow):
		if self.cuda:
			self.cudagl.initjacobian(y_im, y_flow)
			#self.cudagl.initjacobian_CPU(y_im, y_flow)
		else:
			self.cudagl.initjacobian_CPU(y_im, y_flow)

	def jz(self):
		#Compare both and see if they're always off, or just sometimes...

		if self.cuda:
			#print 'jz(). Using GPU (CUDA)'
			jz_GPU = self.cudagl.jz()
			#jz_CPU = self.cudagl.jz_CPU()
			#print 'GPU:', jz_GPU, 'CPU:', jz_CPU
			return jz_GPU

			#return self.cudagl.jz()
		else:
			#print 'Using CPU'
			return self.cudagl.jz_CPU()

	def j(self, state, deltaX, i, j):
		if self.cuda:
			#print 'j(). Using GPU (CUDA)'
			j_GPU = self.cudagl.j(state, deltaX, i, j)
			#j_CPU = self.cudagl.j_CPU(state, deltaX, i, j)
			#print 'GPU:', j_GPU, 'CPU:', j_CPU
			return j_GPU

			#return self.cudagl.j(state, deltaX, i, j)
		else:
			return self.cudagl.j_CPU(state, deltaX, i, j)

class VideoStream:
	def __init__(self, fn, threshold):
		print("Creating video stream from " + fn)
		print("Background subtraction below intensity " + str(threshold))
		self.threshold = threshold
		self.cap = cv2.VideoCapture(fn)
		ret, frame = self.cap.read()
		assert ret, "Cannot open %s" % fn 
		nx,ny = frame.shape[0:2]
		self.nx = nx 
		self.ny = ny
		self.frame = frame
		self.frame_orig = frame.copy()
		self.grayframe = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)

	def read(self, backsub = True):
		ret, frame = self.cap.read()
		self.frame = frame 
		self.grayframe = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		if not backsub:
			return ret, self.frame, self.grayframe
		else:
			(mask, ctrs, fd) = findObjectThreshold(self.frame, threshold = self.threshold)
			#Apply mask
			backframe = np.multiply(np.dstack((mask, mask, mask)), self.frame)
			backgrayframe = np.multiply(mask, self.grayframe)		
			return ret, backframe, backgrayframe 

	def current_frame(self, backsub = True):
		if backsub:
			(mask, ctrs, fd) = findObjectThreshold(self.frame, threshold = self.threshold)
			frame = np.multiply(np.dstack((mask, mask, mask)), self.frame)
		else:
			frame = self.frame 
		return frame 

	def gray_frame(self, backsub = True):
		if backsub:
			(mask, ctrs, fd) = findObjectThreshold(self.frame, threshold = self.threshold)
			frame = np.multiply(mask, self.grayframe)
		else:
			frame = self.grayframe 
		return frame 

	def backsub(self, im = None):
		(mask, ctrs, fd) = findObjectThreshold(self.frame, threshold = self.threshold)
		if im is None:
			return mask, ctrs, fd
		else:
			if len(im.shape) == 2:
				return np.multiply(mask, im)
			#Probably a better way to do this...
			elif im.shape[2] == 2:
				return np.multiply(np.dstack((mask, mask)), im)
			elif im.shape[2] == 3:
				return np.multiply(np.dstack((mask, mask, mask)), im)

	def isOpened(self):
		return self.cap.isOpened()

def readFileToMat(path): 
	from ctypes import sizeof, c_int, c_float, c_double
	sz_int = sizeof(c_int)
	sz_float = sizeof(c_float)
	sz_double = sizeof(c_double)

	import struct 
	#Read binary .mat file, saved using optical_flow_ext.cpp:writeMatToFile()
	fo = open(path, 'rw+')
	arrayType = struct.unpack('i', fo.read(sz_int))[0]
	matWidth = struct.unpack('i', fo.read(sz_int))[0]
	matHeight = struct.unpack('i', fo.read(sz_int))[0]
			
	#FLOAT ONE CHANNEL
	if arrayType == cv2.CV_32F:
		mat = np.zeros((matHeight, matWidth), dtype = np.float32)
		print "Reading CV_32F image"
		for i in range(matHeight):
			for j in range(matWidth):
				val = struct.unpack('f', fo.read(sz_int))[0]
				#if val > 0:
				#	print val 
				mat[i,j] = val
	#DOUBLE ONE CHANNEL
	elif arrayType == cv2.CV_64F:
		mat = np.zeros((matHeight, matWidth), dtype = np.float64)
		print "Reading CV_64F image"
		for i in range(matHeight):
			for j in range(matWidth):
				val = struct.unpack('d', fo.read(sz_double))[0]
				mat[i,j] = val
	#FLOAT THREE CHANNELS
	elif arrayType == cv2.CV_32FC3:
		mat = np.zeros((matHeight, matWidth), dtype = np.float32);
		print "Reading CV_32FC3 image"
		for i in range(matHeight):
			for j in range(matWidth):
				val = struct.unpack('f', fo.read(sz_float))[0]
				mat[i,j] = val
	#DOUBLE THREE CHANNELS
	elif arrayType == cv2.CV_64FC3:
		mat = np.zeros((matHeight, matWidth), dtype = np.float64)
		print "Reading CV_64FC3 image"
		for i in range(matHeight):
			for j in range(matWidth):
				val = struct.unpack('d', fo.read(sz_double))[0]
				mat[i,j] = val
	else:
		print "Error: wrong Mat type: must be CV_32F, CV_64F, CV_32FC3 or CV_64FC3"

	fo.close()
	return mat

def type2str(type):
	#depth = chr(type & CV_MAT_DEPTH_MASK)
	#chans = chr(1 + (type >> CV_CN_SHIFT))
	depth = (type & CV_MAT_DEPTH_MASK)
	chans = (1 + (type >> CV_CN_SHIFT))
	if depth == cv2.CV_8U:
		r = "8U"
	elif depth == cv2.CV_8S:
		r = "8S"
	elif depth == cv2.CV_16U:
		r = "16U"
	elif depth == cv2.CV_16S:
		r = "16S"
	elif depth == cv2.CV_32S:
		r = "32S"
	elif depth == cv2.CV_32F:
		r = "32F"
	elif depth == cv2.CV_64F:
		r = "64F"
	else:
		r = "User"
	r += "C"
	r += (str(chans)+'0')
	return r