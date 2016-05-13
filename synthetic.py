"""Generate synthetic datasets"""

def test_data(nx, ny):
	nframes = 10
	speed = 3
	video = np.zeros((nx, ny, nframes), dtype = np.uint8)
	im = np.zeros((nx, ny))
	#Set up a box in the first frame, with some basic changes in intensity
	start = nx//3
	end = 2*nx//3
	width = end-start
	height = end-start
	flow = np.zeros((nx, ny, 2, nframes), dtype=np.float32)

	for i in range(start,end):
		for j in range(start,end):
			if i > j:
				col = 128
			else:
				col = 255
			im[i,j] = col
	im = np.flipud(im)
	#Translate the box for a few frames
	for i in range(nframes):
		imtrans = im[speed*i:,speed*i:]
		s = start-speed*i
		flow[s:s+width, s:s+height,0,i] = -speed
		flow[s:s+width, s:s+height,1,i] = -speed
		#flow[s:s+width, s:s+height,0,i] = -speed
		#flow[s:s+width, s:s+height,1,i] = -speed
		if i > 0:
			video[:-speed*i,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans
	return video, flow

def test_data_up(nx, ny):
	nframes = 30
	speed = 2
	video = np.zeros((nx, ny, nframes), dtype = np.uint8)
	im = np.zeros((nx, ny))
	#Set up a box in the first frame, with some basic changes in intensity
	start = nx//3
	end = 2*nx//3
	for i in range(start,end):
		for j in range(start,end):
			if i > j:
				col = 128
			else:
				col = 255
			im[i,j] = col
	#Translate the box for a few frames
	for i in range(nframes):
		imtrans = im[:,speed*i:]
		if i > 0:
			video[:,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans
	return video 

def test_data_texture(nx, ny):
	noise = 0
	nframes = 10
	speed = 3
	video = np.zeros((nx, ny, nframes), dtype = np.uint8)
	im = np.zeros((nx, ny))
	#Set up a box in the first frame, with some basic changes in intensity
	start = nx//3
	end = 2*nx//3
	for i in range(start,end):
		for j in range(start,end):
			if i > j:
				col = 128
			else:
				col = 200
			im[i,j] = col + noise*np.random.normal(size = (1,1))
	#Add noise
	#noise = 
	#Apply Gaussian blur 
	#im = im + noise 
	im = cv2.GaussianBlur(im,(15,15),0)
	#Translate the box for a few frames
	for i in range(nframes):
		imtrans = im[speed*i:,speed*i:]
		if i > 0:
			video[:-speed*i,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans
	return video 

def test_data_image(fn = './video/milkyway.jpg'):
	imsrc = cv2.imread(fn,0)
	nx = imsrc.shape[0]
	ny = imsrc.shape[1]
	im = np.zeros((nx, ny), dtype=np.uint8)
	imsrc = cv2.resize(imsrc,None,fx=1./2, fy=1./2, interpolation = cv2.INTER_CUBIC)
	width = imsrc.shape[0]
	height = imsrc.shape[1]
	start = nx//3
	im[start:start+width, start:start+height] = imsrc

	nframes = 10
	speed = 3
	noise = 10
	video = np.zeros((nx, ny, nframes), dtype = np.uint8)
	flow = np.zeros((nx, ny, 2, nframes), dtype=np.float32)

	#Translate the box for a few frames
	for i in range(nframes):
		start = nx//3-i*speed
		imtrans = im[speed*i:,speed*i:]
		if i > 0:
			flow[start:start+width, start:start+height,0,i] = -speed
			flow[start:start+width, start:start+height,1,i] = -speed
			video[:-speed*i,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans

	imnoisex = np.zeros((nx, ny))
	imnoisey = np.zeros((nx, ny))
	#Set up a box in the first frame, with some basic changes in intensity
	start = nx//3
	end = nx//3+width
	for i in range(start,end):
		for j in range(start,end):
			imnoisex[i,j] = noise*np.random.normal(size = (1,1))
			imnoisey[i,j] = noise*np.random.normal(size = (1,1))
	#Add noise
	#noise = 
	#Apply Gaussian blur 
	#im = im + noise 
	imnoisex = cv2.GaussianBlur(imnoisex,(15,15),0)
	imnoisey = cv2.GaussianBlur(imnoisey,(15,15),0)

	flow[:,:,0,0] = imnoisex
	flow[:,:,0,0] = imnoisey

	return video, flow 