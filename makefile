useLapack = false

lapackLink = `pkg-config blas lapack --cflags --libs`

LDFLAGS = '-Wl,-rpath,$$ORIGIN/../lib' -Llib/

PYTHON_INCLUDES = `python3 -m pybind11 --includes` 

NUMPY_INCLUDES = `python3 -c 'import numpy; print(numpy.get_include())'`

PY_SUFFIX := $(shell python3-config --extension-suffix)

DEBUG_FLAGS = -g -Wall -Wextra

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
	g++ $(DEBUG_FLAGS) --std=c++11 -O3 Mat_test/Mat_test.cpp $(LDFLAGS) -linverseLapack -o bin/Mat_test $(lapackLink)
else	
Mat_test: Mat_test/Mat_test.cpp include/Mat.h include/Mat_Math.h lib/libinverse.so
	g++ $(DEBUG_FLAGS) --std=c++11 -O3 Mat_test/Mat_test.cpp $(LDFLAGS) -linverse -o bin/Mat_test
endif

Flood_Fill: Flood_Fill/Flood_Fill.cpp include/Mat.h
	g++ $(DEBUG_FLAGS) --std=c++11 -O3 Flood_Fill/Flood_Fill.cpp -o bin/Flood_Fill

Flood_Pybind: Mat_Pybind Flood_Fill
	g++ -O3 -Wall -shared -std=c++14 -fPIC -I include $(PYTHON_INCLUDES) -I $(NUMPY_INCLUDES) Flood_Fill/Flood_Fill.cpp Pybind/Flood_Fill_Pybind.cpp -o Python/Flood_Pybind$(PY_SUFFIX)
	
Mat_Pybind: Python/Mat_Pybind$(PY_SUFFIX)

Python/Mat_Pybind$(PY_SUFFIX): Pybind/Mat_Pybind.cpp include/Mat.h include/Matmodule.h
	g++ -O3 -Wall -shared -std=c++14 -fPIC -I include $(PYTHON_INCLUDES) -I $(NUMPY_INCLUDES) Pybind/Mat_Pybind.cpp -o Python/Mat_Pybind$(PY_SUFFIX)

Mat_Debug: Python/Mat_Debug$(PY_SUFFIX)
Python/Mat_Debug$(PY_SUFFIX): Pybind/Mat_debug_bindings.cpp include/Mat.h include/Matmodule.h
	g++ -O3 -Wall -shared -std=c++14 -fPIC -I include $(PYTHON_INCLUDES) -I $(NUMPY_INCLUDES) Pybind/Mat_debug_bindings.cpp -o Python/Mat_Debug$(PY_SUFFIX)
	