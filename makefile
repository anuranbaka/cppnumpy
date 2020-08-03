all: Mat_test Flood_test

install: all #We don't install this. Just build it.

Mat_test: Mat_test.cpp projects/matlib/Mat.h
	g++ -g --std=c++11 -O3 Mat_Test/Mat_test.cpp -o Mat_Test/Mat_test

Flood_test: projects/Flood_Fill/Flood_test.cpp projects/Flood_Fill/floodFill.h projects/matlib/Mat.h
	g++ -g --std=c++11 -O3 Flood_Fill/Flood_test.cpp -o Flood_Fill/Flood_test