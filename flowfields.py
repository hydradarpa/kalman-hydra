flowfields = {}
#Velocity fields:
flowfields['translate_leftup'] = lambda pt,t: (-3, -3)
flowfields['translate_leftup_stretch'] = lambda pt,t: (-3+pt[0]/200., -3+pt[1]/200.)
