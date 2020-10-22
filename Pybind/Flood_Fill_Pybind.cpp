#include "../Flood_Fill/Flood_Fill.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Mat_Pybind.h>

template<class T>
Mat<T> floodFillPy(Mat<T>& image, vector<size_t> start, T color, int connectivity = 4){
    floodFill(image, start, color, connectivity);
    return image;
}
PYBIND11_MODULE(Flood_Pybind, m){
    m.doc() = "performs floodFill in c++";
    m.def("floodFill", &floodFillPy<double>,
        "performs floodFill on a given point in a matrix");
}