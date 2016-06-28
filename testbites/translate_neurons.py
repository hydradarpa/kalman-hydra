fn_in = './synthetictests/hydra_neurons1_neurons.csv'
fn_out = './synthetictests/hydra_neurons1a_neurons.csv'
f_out = open(fn_out, 'w')
x0 = 232
y0 = 157
s = 0.6
for line in open(fn_in):
	words = line.split(',')
	n = words[0]
	x = float(words[1])
	y = float(words[2])
	xp = s*(x+x0)
	yp = s*(y+y0)
	f_out.write("%s,%f,%f\n"%(n,xp,yp))
f_out.close()