useLapack = false

sim32bit = false

lapackLink = `pkg-config blas lapack --cflags --libs`

LDFLAGS = '-Wl,-rpath,$$ORIGIN/../lib' -Llib/

INCLUDES = include

PYTHON_INCLUDES = `python3 -m pybind11 --includes` 

NUMPY_INCLUDES = `python3 -c 'import numpy; print(numpy.get_include())'`

PY_SUFFIX := $(shell python3-config --extension-suffix)

DEBUG_FLAGS = -g -Wall -Wextra -pedantic

all: bin/matTest floodPybind matDebug bin/errorTest | lib bin

lib:
	mkdir lib/

bin:
	mkdir bin/

install:
	install -d $(DESTDIR)$(PREFIX)/lib/
	install lib/lib* $(PREFIX)/lib
	install -d $(DESTDIR)$(PREFIX)/include/
	install include/* $(PREFIX)/include

clean:
	rm ./lib/*.so ./bin/* python/*.so

lib/libInverseLapack.so: src/matMathLapack.cpp | lib
	g++ -g --std=c++11 -O3 -fPIC -I $(INCLUDES) src/matMathLapack.cpp -shared -o lib/libInverseLapack.so

lib/libInverse.so: src/matMath.cpp | lib
	g++ -g --std=c++11 -O3 -fPIC -I $(INCLUDES) src/matMath.cpp -shared -o lib/libInverse.so

lib/libInverseLapack32.so: src/matMathLapack.cpp | lib
	g++ -g --std=c++11 -O3 -fPIC -m32 -I $(INCLUDES) src/matMathLapack.cpp -shared -o lib/libInverseLapack32.so

lib/libInverse32.so: src/matMath.cpp | lib
	g++ -g --std=c++11 -O3 -fPIC -m32 -I $(INCLUDES) src/matMath.cpp -shared -o lib/libInverse32.so

ifeq ($(sim32bit),true)
ifeq ($(useLapack),true)
bin/matTest: matTest/matTest.cpp include/Mat.h include/matMath.h lib/libInverseLapack32.so | bin
	g++ $(DEBUG_FLAGS) --std=c++11 -I $(INCLUDES) -m32 matTest/matTest.cpp $(LDFLAGS) -lInverseLapack32 -o bin/matTest $(lapackLink)
else	
bin/matTest: matTest/matTest.cpp include/Mat.h include/matMath.h lib/libInverse32.so | bin
	g++ $(DEBUG_FLAGS) --std=c++11 -I $(INCLUDES) -m32 matTest/matTest.cpp $(LDFLAGS) -lInverse32 -o bin/matTest
endif
else
ifeq ($(useLapack),true)
bin/matTest: matTest/matTest.cpp include/Mat.h include/matMath.h lib/libInverseLapack.so | bin
	g++ $(DEBUG_FLAGS) --std=c++11 -I $(INCLUDES) matTest/matTest.cpp $(LDFLAGS) -lInverseLapack -o bin/matTest $(lapackLink)
else	
bin/matTest: matTest/matTest.cpp include/Mat.h include/matMath.h lib/libInverse.so | bin
	g++ $(DEBUG_FLAGS) --std=c++11 -I $(INCLUDES) matTest/matTest.cpp $(LDFLAGS) -lInverse -o bin/matTest
endif
endif

bin/errorTest: matTest/errorTest.cpp include/Mat.h | bin
	g++ $(DEBUG_FLAGS) --std=c++11 -I $(INCLUDES) matTest/errorTest.cpp -o bin/errorTest

bin/floodFill: floodFill/floodFill.cpp include/Mat.h include/floodFill.h | bin
	g++ $(DEBUG_FLAGS) --std=c++11 -O3 -I $(INCLUDES) floodFill/floodFill.cpp -o bin/floodFill

floodPybind: python/floodPybind$(PY_SUFFIX)
python/floodPybind$(PY_SUFFIX): pybind/floodFillPybind.cpp include/matPybind.h bin/floodFill
	g++ -O3 -Wall -shared -std=c++14 -fPIC -I include $(PYTHON_INCLUDES) -I $(NUMPY_INCLUDES) floodFill/floodFill.cpp pybind/floodFillPybind.cpp -o python/floodPybind$(PY_SUFFIX)

matDebug: python/matDebug$(PY_SUFFIX)
python/matDebug$(PY_SUFFIX): pybind/matDebugTest.cpp include/Mat.h include/pythonAPI.h
	g++ -O3 -Wall -shared -std=c++14 -fPIC -I include $(PYTHON_INCLUDES) -I $(NUMPY_INCLUDES) pybind/matDebugTest.cpp -o python/matDebug$(PY_SUFFIX)
	
python/pyCapsuleTest: include/matPybind.h pybind/pyCapsuleTest.cpp
	g++ $(DEBUG_FLAGS) -shared -std=c++14 -fPIC -I include $(PYTHON_INCLUDES) -I $(NUMPY_INCLUDES) pybind/pyCapsuleTest.cpp -o python/pyCapsuleTest$(PY_SUFFIX)