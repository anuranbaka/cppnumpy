useLapack = false

lapackLink = `pkg-config blas lapack --cflags --libs`

all: Mat_test Flood_test

install: all #We don't install this. Just build it.

clean:
	rm ./lib/*.o ./bin/*

lib/inverseLapack.o: lib/inverseLapack.cpp
	g++ -g --std=c++11 -O3 src/inverseLapack.cpp -c -o lib/inverseLapack.o

ifeq ($(useLapack),true)
Mat_test: Mat_test/Mat_test.cpp include/Mat.h include/Mat_Math.h src/inverseLapack.o
	g++ -g --std=c++11 -O3 src/inverseLapack.o Mat_test/Mat_test.cpp -o bin/Mat_test $(lapackLink)
else	
Mat_test: Mat_test/Mat_test.cpp include/Mat.h include/Mat_Math.h src/inverse.o
	g++ -g --std=c++11 -O3 src/inverse.o Mat_test/Mat_test.cpp -o bin/Mat_test
endif

Flood_test: Flood_Fill/Flood_test.cpp include/Mat.h
	g++ -g --std=c++11 -O3 Flood_Fill/Flood_test.cpp -o bin/Flood_test