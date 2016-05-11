CC=g++
CFLAGS=`pkg-config --cflags opencv`
LIBS=`pkg-config --libs opencv`

all: opticflow

opticflow:
	$(CC) $(CLFAGS) optical_flow_ext.cpp $(LIBS) -o bin/optical_flow_ext
