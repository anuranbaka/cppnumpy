#include "../Flood_Fill/Flood_Fill.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(Flood_Pybind, m){
    m.doc() = "performs floodFill in c++";
    m.def("floodFill", &floodFill<double>,
        "performs floodFill on a given point in a matrix");
}