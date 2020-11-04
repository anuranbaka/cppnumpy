#include "../floodFill/floodFill.h" //header containing code to be bound
#include <pybind11/pybind11.h> //pybind11 
#include <matPybind.h> //allows conversion of pyarray->Mat
#include <pybind11/stl.h> //allows conversion of std::vector
#include <pybind11/functional.h> //allows passing of function pointers

template<class T>
void floodFillPy(Mat<T>& image, vector<size_t> start, T color, int connectivity = 4){
    floodFill(image, start, color, connectivity);
    return;
}

PYBIND11_MODULE(FloodPybind, m){
    m.doc() = "performs floodFill in c++";
    m.def("floodFill", &floodFillPy<bool>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<short>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<unsigned short>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<int>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<unsigned int>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<long>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<unsigned long>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<long long>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<unsigned long long>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<signed char>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<unsigned char>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<char>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<float>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFill", &floodFillPy<double>);
}