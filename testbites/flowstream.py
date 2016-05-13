from renderer import FlowStream
from cvtools import *
self = FlowStream('./video/johntest_brightcontrast_short/flow')
self.frame = 144
fn_x = self.path + ("_%03d"%self.frame) + "_x.mat"
fn_y = self.path + ("_%03d"%self.frame) + "_y.mat"
self.frame += 1
self.flowx = readFileToMat(fn_x)
self.flowy = readFileToMat(fn_y)

self.draw()