flowfields = {}
#Velocity fields:
flowfields['translate_leftup'] = lambda pt,t: (-3., -3.)
flowfields['translate_leftup_stretch'] = lambda pt,t: (-3.+pt[0]/200., -3.+pt[1]/200.)
y0 = 500.
x0 = 500.
flowfields['rotate'] = lambda pt,t: (-(pt[1]-y0)/100., (pt[0]-x0)/100.)
flowfields['warp'] = lambda pt,t: (-(pt[0]/500.-1.)*pt[0]*(pt[0]/1000.-1.)/100.,(pt[1]/500.-1.)*pt[1]*(pt[1]/1000.-1.)/100.)