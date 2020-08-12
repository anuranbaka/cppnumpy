all: Mat_test Flood_test

install: all #We don't install this. Just build it.

Mat_test: Mat_test.cpp projects/matlib/Mat.h projects/Mat_Math/inverse.o
	g++ -g --std=c++11 -O3 projects/Mat_Math/inverse.o Mat_test.cpp -o Mat_test

Flood_test: projects/Flood_Fill/Flood_test.cpp projects/Flood_Fill/floodFill.h projects/matlib/Mat.h
	g++ -g --std=c++11 -O3 projects/Flood_Fill/Flood_test.cpp -o projects/Flood_Fill/Flood_test