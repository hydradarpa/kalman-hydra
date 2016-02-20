CC=g++
CFLAGS=`pkg-config --cflags opencv`
LIBS=`pkg-config --libs opencv`

all: video opticflow write

video:
	$(CC) $(CFLAGS) starter_video.cpp $(LIBS) -o starter_video

opticflow:
	$(CC) $(CFLAGS) optical_flow.cpp $(LIBS) -o optical_flow

flowvideo:
	$(CC) $(CLFAGS) optical_flow_video.cpp $(LIBS) -o optical_flow_video

write:
	$(CC) $(CFLAGS) video-write.cpp $(LIBS) -o video-write

write2:
	$(CC) $(CFLAGS) testvidwriter2.cpp $(LIBS) -o video-write2

hdf:
	$(CC) $(CFLAGS) testhdf.cpp $(LIBS) -o testhdf