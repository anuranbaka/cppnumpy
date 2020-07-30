all: Mat_test Flood_test

install: all #We don't install this. Just build it.

Mat_test: Mat_Test/Mat_test.cpp matlib/Mat.h Mat_Math/inverse.o
	g++ -g --std=c++11 -O3 Mat_Math/inverse.o Mat_Test/Mat_test.cpp -o Mat_Test/Mat_test

Flood_test: Flood_Fill/Flood_test.cpp Flood_Fill/floodFill.h matlib/Mat.h
	g++ -g --std=c++11 -O3 Flood_Fill/Flood_test.cpp -o Flood_Fill/Flood_test