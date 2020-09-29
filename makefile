useLapack = false

lapackLink = `pkg-config blas lapack --cflags --libs`

LDFLAGS = '-Wl,-rpath,$$ORIGIN/../lib' -Llib/

PYTHON_INCLUDES = `python3 -m pybind11 --includes` 

NUMPY_INCLUDES = `python3 -c 'import numpy; print(numpy.get_include())'`

all: Mat_test Flood_Fill Mat_Pybind

install: all
	install -d $(DESTDIR)$(PREFIX)/lib/
	install lib/lib* $(PREFIX)/lib
	install -d $(DESTDIR)$(PREFIX)/include/
	install include/* $(PREFIX)/include

clean:
	rm ./lib/*.o ./bin/*

bin/libinverseLapack.so: src/inverseLapack.cpp
	g++ -g --std=c++11 -O3 -fPIC src/inverseLapack.cpp -shared -o lib/libinverseLapack.so

bin/libinverse.so: src/inverse.cpp
	g++ -g --std=c++11 -O3 -fPIC src/inverse.cpp -shared -o lib/libinverse.so

ifeq ($(useLapack),true)
Mat_test: Mat_test/Mat_test.cpp include/Mat.h include/Mat_Math.h lib/libinverseLapack.so
	g++ -g --std=c++11 -O3 Mat_test/Mat_test.cpp $(LDFLAGS) -linverseLapack -o bin/Mat_test $(lapackLink)
else	
Mat_test: Mat_test/Mat_test.cpp include/Mat.h include/Mat_Math.h lib/libinverse.so
	g++ -g --std=c++11 -O3 Mat_test/Mat_test.cpp $(LDFLAGS) -linverse -o bin/Mat_test
endif

Flood_Fill: Flood_Fill/Flood_Fill.cpp include/Mat.h
	g++ -g --std=c++11 -O3 Flood_Fill/Flood_Fill.cpp -o bin/Flood_Fill
	
Mat_Pybind: Pybind/Mat_Pybind.cpp
	g++ -O3 -Wall -shared -std=c++14 -fPIC -I include $(PYTHON_INCLUDES) -I $(NUMPY_INCLUDES) Pybind/Mat_Pybind.cpp -o Pybind/Mat_Pybind`python3-config --extension-suffix`
	