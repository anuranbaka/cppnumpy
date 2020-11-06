useLapack = false

lapackLink = `pkg-config blas lapack --cflags --libs`

LDFLAGS = '-Wl,-rpath,$$ORIGIN/../lib' -Llib/

PYTHON_INCLUDES = `python3 -m pybind11 --includes` 

NUMPY_INCLUDES = `python3 -c 'import numpy; print(numpy.get_include())'`

PY_SUFFIX := $(shell python3-config --extension-suffix)

DEBUG_FLAGS = -g -Wall -Wextra

all: matTest floodPybind

install: all
	install -d $(DESTDIR)$(PREFIX)/lib/
	install lib/lib* $(PREFIX)/lib
	install -d $(DESTDIR)$(PREFIX)/include/
	install include/* $(PREFIX)/include

clean:
	rm ./lib/*.so ./bin/* python/*.so

lib/libInverseLapack.so: src/matMathLapack.cpp
	g++ -g --std=c++11 -O3 -fPIC src/matMathLapack.cpp -shared -o lib/libInverseLapack.so

lib/libInverse.so: src/matMath.cpp
	g++ -g --std=c++11 -O3 -fPIC src/matMath.cpp -shared -o lib/libInverse.so

ifeq ($(useLapack),true)
matTest: matTest/matTest.cpp include/Mat.h include/matMath.h lib/libInverseLapack.so
	g++ $(DEBUG_FLAGS) --std=c++11 -O3 matTest/matTest.cpp $(LDFLAGS) -lInverseLapack -o bin/matTest $(lapackLink)
else	
matTest: matTest/matTest.cpp include/Mat.h include/matMath.h lib/libInverse.so
	g++ $(DEBUG_FLAGS) --std=c++11 -O3 matTest/matTest.cpp $(LDFLAGS) -lInverse -o bin/matTest
endif

floodFill: floodFill/floodFill.cpp include/Mat.h
	g++ $(DEBUG_FLAGS) --std=c++11 -O3 floodFill/floodFill.cpp -o bin/floodFill

floodPybind: include/matPybind.h floodFill
	g++ -O3 -Wall -shared -std=c++14 -fPIC -I include $(PYTHON_INCLUDES) -I $(NUMPY_INCLUDES) floodFill/floodFill.cpp pybind/floodFillPybind.cpp -o python/floodPybind$(PY_SUFFIX)

matDebug: python/matDebug$(PY_SUFFIX)
python/matDebug$(PY_SUFFIX): pybind/matDebugBindings.cpp include/Mat.h include/pythonAPI.h
	g++ -O3 -Wall -shared -std=c++14 -fPIC -I include $(PYTHON_INCLUDES) -I $(NUMPY_INCLUDES) pybind/matDebugBindings.cpp -o python/matDebug$(PY_SUFFIX)
	