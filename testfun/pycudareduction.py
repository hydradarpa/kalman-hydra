from pycuda.reduction import ReductionKernel
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy

a = gpuarray.arange(400, dtype=numpy.float32)
b = gpuarray.arange(400, dtype=numpy.float32)

dot = ReductionKernel(dtype_out=numpy.float32, neutral="0",reduce_expr="a+b", map_expr="x[i]*y[i]",arguments="const float *x, const float *y")

a_dot_b = dot(a, b).get()
a_dot_b_cpu = numpy.dot(a.get(), b.get())