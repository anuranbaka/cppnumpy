#include <floodFill.h> //header containing code to be bound
#include <pybind11/pybind11.h> //pybind11 
#include <matPybind.h> //allows conversion of pyarray->Mat
#include <pybind11/stl.h> //allows conversion of std::vector

PYBIND11_MODULE(floodPybind, m){
    m.doc() = "performs floodFill in c++";
    m.def("floodFill", &floodFill<bool>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<short>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<unsigned short>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<int>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<unsigned int>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<long>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<unsigned long>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<long long>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<unsigned long long>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<signed char>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<unsigned char>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<char>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<float>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFill<double>,
        "performs floodFill on a given point in a matrix");
}