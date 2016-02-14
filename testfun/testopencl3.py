# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
# PyOpenCL imports
import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties
import sys 
import numpy as np 

# OpenCL kernel that generates a sine function.
clkernel = """
__kernel void clkernel(__global float2* clpos, __global float2* glpos)
{
    //get our index in the array
    unsigned int i = get_global_id(0);

    // copy the x coordinate from the CL buffer to the GL buffer
    glpos[i].x = clpos[i].x;

    // calculate the y coordinate and copy it on the GL buffer
    glpos[i].y = 0.5 * sin(10.0 * clpos[i].x);
}
"""

class TestWindow(QtGui.QMainWindow):
    def __init__(self):
        super(TestWindow, self).__init__()
        # generate random data points
        self.data = np.zeros((10000,2))
        self.data[:,0] = np.linspace(-1.,1.,len(self.data))
        self.data = np.array(self.data, dtype=np.float32)
        # initialize the GL widget
        self.widget = GLPlotWidget()
        self.widget.set_data(self.data)
        # put the window at the screen position (100, 100)
        self.setGeometry(100, 100, self.widget.width, self.widget.height)
        self.setCentralWidget(self.widget)
        self.show()

# create the Qt App and window
app = QtGui.QApplication(sys.argv)
window = TestWindow()
window.show()
app.exec_()

def clinit():
    """Initialize OpenCL with GL-CL interop.
    """
    plats = cl.get_platforms()
    # handling OSX
    if sys.platform == "darwin":
        ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                             devices=[])
    else:
        c = [(cl.context_properties.PLATFORM, plats[0])] + get_gl_sharing_context_properties()
        #c = [get_gl_sharing_context_properties()[1]]
        #c = [(cl.context_properties.PLATFORM, plats[0])]
        ctx = cl.Context(properties=c, devices=None)
    queue = cl.CommandQueue(ctx)
    return ctx, queue


# empty OpenGL VBO
data = np.array([0, 1])
glbuf = glvbo.VBO(data=np.zeros(data.shape),
					   usage=gl.GL_DYNAMIC_DRAW,
					   target=gl.GL_ARRAY_BUFFER)
glbuf.bind()
# initialize the CL context
ctx, queue = clinit()
# create a pure read-only OpenCL buffer
clbuf = cl.Buffer(ctx,
					cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
					hostbuf=data)
# create an interop object to access to GL VBO from OpenCL
glclbuf = cl.GLBuffer(ctx, cl.mem_flags.READ_WRITE,
					int(glbuf.buffers[0]))
# build the OpenCL program
program = cl.Program(ctx, clkernel).build()
# release the PyOpenCL queue
queue.finish()