CC=g++
CFLAGS=`pkg-config --cflags opencv`
LIBS=`pkg-config --libs opencv`

all: opticflow deepflow simpleflow

opticflow:
	$(CC) $(CLFAGS) optical_flow_ext.cpp $(LIBS) -o bin/optical_flow_ext

deepflow:
	$(CC) $(CLFAGS) deep_optical_flow_ext.cpp $(LIBS) -o bin/deep_optical_flow_ext

simpleflow:
	$(CC) $(CLFAGS) simple_optical_flow_ext.cpp $(LIBS) -o bin/simple_optical_flow_ext
