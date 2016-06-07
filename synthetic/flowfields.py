flowfields = {}
#Velocity fields:
flowfields['translate_leftup'] = lambda pt,t: (-1.5, -1.5)
flowfields['translate_leftup_stretch'] = lambda pt,t: (-1.+pt[0]/200., -1.+pt[1]/200.)
y0 = 300.
x0 = 300.
flowfields['rotate'] = lambda pt,t: (-(pt[1]-y0)/25., (pt[0]-x0)/25.)
flowfields['warp'] = lambda pt,t: (-(pt[0]/300.-1.)*pt[0]*(pt[0]/600.-1.)/75.,(pt[1]/300.-1.)*pt[1]*(pt[1]/600.-1.)/75.)