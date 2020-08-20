useLapack = false

lapackLink = `pkg-config blas lapack --cflags --libs`

all: Mat_test Flood_test

install: all #We don't install this. Just build it.

clean:
	rm ./projects/Mat_Math/*.o Mat_test Flood_test

projects/Mat_Math/inverseLapack.o: projects/Mat_Math/inverseLapack.cpp
	g++ -g --std=c++11 -O3 projects/Mat_Math/inverseLapack.cpp -c -o projects/Mat_Math/inverseLapack.o

ifeq ($(useLapack),true)
Mat_test: Mat_test.cpp projects/matlib/Mat.h projects/Mat_Math/inverseLapack.o
	g++ -g --std=c++11 -O3 projects/Mat_Math/inverseLapack.o Mat_test.cpp -o Mat_test $(lapackLink) 
else	
Mat_test: Mat_test.cpp projects/matlib/Mat.h projects/Mat_Math/inverse.o
	g++ -g --std=c++11 -O3 projects/Mat_Math/inverse.o Mat_test.cpp -o Mat_test
endif

Flood_test: projects/Flood_Fill/Flood_test.cpp projects/Flood_Fill/floodFill.h projects/matlib/Mat.h
	g++ -g --std=c++11 -O3 projects/Flood_Fill/Flood_test.cpp -o Flood_test