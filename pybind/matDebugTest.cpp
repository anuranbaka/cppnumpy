#include <matDebugTest.h> //header containing code to be bound
#include <pybind11/pybind11.h> //pybind11 
#include <matPybind.h> //allows conversion of pyarray->Mat

PYBIND11_MODULE(matDebug, m){
    m.def("copyTest", &copyTest<double>,
        "returns a copy of the current matrix");
    m.def("copyTest", &copyTest<int>,
        "returns a copy of the current matrix");
    m.def("copyTest", &copyTest<long long>,
        "returns a copy of the current matrix");
}
