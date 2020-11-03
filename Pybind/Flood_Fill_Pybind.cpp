#include "../Flood_Fill/Flood_Fill.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Mat_Pybind.h>

template<class T>
Mat<T> floodFillPy(Mat<T>& image, vector<size_t> start, T color, int connectivity = 4){
    floodFill(image, start, color, connectivity);
    return image;
}
/*template<class T>
Mat<T> floodCustomPy(Mat<T>& image, vector<size_t> start, T color, bool (*func)(T,T), int connectivity = 4){
    floodFillCustom(image, start, color, func, connectivity);
    return image;
}*/
PYBIND11_MODULE(Flood_Pybind, m){
    m.doc() = "performs floodFill in c++";
    //m.def("floodFill", &floodFillPy<bool>,
    //    "performs floodFill on a given point in a matrix");
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
    m.def("floodFill", &floodFillPy<double>,
        "performs floodFill on a given point in a matrix");
    /*m.def("floodFillCustom", &floodCustomPy<bool>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<short>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<unsigned short>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<int>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<unsigned int>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<long>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<unsigned long>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<long long>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<unsigned long long>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<signed char>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<unsigned char>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<char>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<float>,
        "performs floodFill on a given point in a matrix");
    m.def("floodFillCustom", &floodCustomPy<double>,
        "performs floodFill on a given point in a matrix");*/
}