#include <pybind11/pybind11.h>
#include <Mat.h>

namespace py = pybind11;

template <class T>
void declare_debug(py::module &m, const std::string &typestr){
    std::string pyclass_name = std::string("Mat") + typestr;
    py::class_<Mat<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<size_t>())
    .def(py::init<size_t, size_t>())
    .def(py::init<std::initializer_list<T>, size_t>())
    .def(py::init<std::initializer_list<T>, size_t, size_t>())
    //iterator/capacity functions
    .def("size", &Mat<T>::size)
    .def("rows", &Mat<T>::rows)
    .def("columns", &Mat<T>::columns)
    .def("inbounds", py::overload_cast<size_t>(&Mat<T>::inbounds), "checks if given coordinates are inbounds")
    //element access/assignment
    .def("__call__", py::overload_cast<size_t>(&Mat<T>::operator()))
    .def("__call__", py::overload_cast<size_t, size_t>(&Mat<T>::operator()))
    .def("assign", py::overload_cast<const Mat<T>&>(&Mat<T>::operator=))
    .def("assign", py::overload_cast<T>(&Mat<T>::operator=))
    //arithmetic operations
    .def("__add__", py::overload_cast<const Mat<T>&>(&Mat<T>::operator+))
    .def("__add__", py::overload_cast<T>(&Mat<T>::operator+))
    .def("__radd__", py::overload_cast<T>(&Mat<T>::operator+))
    .def("__iadd__", py::overload_cast<const Mat<T>&>(&Mat<T>::operator+=))
    .def("__iadd__", py::overload_cast<T>(&Mat<T>::operator+=))
    .def("__sub__", py::overload_cast<const Mat<T>&>(&Mat<T>::operator-))
    .def("__sub__", py::overload_cast<T>(&Mat<T>::operator-))
    .def("__rsub__", [](Mat<T> &a, T b) {
        return b - a;
    }, py::is_operator())
    .def("__isub__", py::overload_cast<const Mat<T>&>(&Mat<T>::operator-=))
    .def("__isub__", py::overload_cast<T>(&Mat<T>::operator-=))
    .def("__mul__", py::overload_cast<const Mat<T>&>(&Mat<T>::operator*))
    .def("__mul__", py::overload_cast<T>(&Mat<T>::operator*))
    .def("__rmul__", py::overload_cast<T>(&Mat<T>::operator*))
    .def("__imul__", py::overload_cast<const Mat<T>&>(&Mat<T>::operator*=))
    .def("__imul__", py::overload_cast<T>(&Mat<T>::operator*=))
    .def("__truediv__", py::overload_cast<const Mat<T>&>(&Mat<T>::operator/))
    .def("__truediv__", py::overload_cast<T>(&Mat<T>::operator/))
    .def("__rtruediv__", [](Mat<T> &a, T b) {
        return b / a;
    }, py::is_operator())
    .def("__itruediv__", py::overload_cast<const Mat<T>&>(&Mat<T>::operator/=))
    .def("__itruediv__", py::overload_cast<T>(&Mat<T>::operator/=))
    .def("__neg__", py::overload_cast<const Mat<T>&>(&Mat<T>::operator-))
    .def("__matmul__", &Mat<T>::operator^)
    .def("T", py::overload_cast<>(&Mat<T>::T), "hard transpose")
    .def("t", &Mat<T>::t)
    //logical operations
    //.def("__and__", py::overload_cast<const Mat<T>&>(&Mat<T>::operator&<T>))
    .def("__and__", (Mat<bool> (Mat<T>::*)(bool))(&Mat<T>::operator&))
    .def("__rand__", (Mat<bool> (Mat<T>::*)(bool))(&Mat<T>::operator&))
    //.def("__or__", py::overload_cast<const Mat<T>&>(&Mat<T>::operator|<T>))
    .def("__or__", (Mat<bool> (Mat<T>::*)(bool))(&Mat<T>::operator|))
    .def("__ror__", (Mat<bool> (Mat<T>::*)(bool))(&Mat<T>::operator|))
    .def("not", &Mat<T>::operator!)
    .def("__eq__", py::overload_cast<T>(&Mat<T>::operator==))
    .def("__eq__", py::overload_cast<const Mat<T>>(&Mat<T>::operator==))
    .def("__eq__", py::overload_cast<T>(&Mat<T>::operator==))
    .def("__ne__", py::overload_cast<const Mat<T>>(&Mat<T>::operator!=))
    .def("__ne__", py::overload_cast<T>(&Mat<T>::operator!=))
    .def("__lt__", py::overload_cast<const Mat<T>>(&Mat<T>::operator<))
    .def("__lt__", py::overload_cast<T>(&Mat<T>::operator<))
    .def("__le__", py::overload_cast<const Mat<T>>(&Mat<T>::operator<=))
    .def("__le__", py::overload_cast<T>(&Mat<T>::operator<=))
    .def("__gt__", py::overload_cast<const Mat<T>>(&Mat<T>::operator>))
    .def("__gt__", py::overload_cast<T>(&Mat<T>::operator>))
    .def("__ge__", py::overload_cast<const Mat<T>>(&Mat<T>::operator>=))
    .def("__ge__", py::overload_cast<T>(&Mat<T>::operator>=))
    .def("all", &Mat<T>::all)
    .def("any", &Mat<T>::any)
    //meta functions
    //.def("broadcast", py::overload_cast<const Mat<T>&, T (*)(T, T)>(&Mat<T>::broadcast<T, T>), "elementwise broadcast function")
    //.def("broadcast", py::overload_cast<T, T (*)(T, T)>(&Mat<T>::broadcast<T, T>), "elementwise broadcast function")
    //.def("broadcast", py::overload_cast<T, Mat<T>&, T (*)(T, T)>(&Mat<T>::broadcast<T, T>), "elementwise broadcast function")
    .def("roi", &Mat<T>::roi)
    .def("print", py::overload_cast<>(&Mat<T>::print), "print contents of matrix")
    //.def("copy", py::overload_cast<>(&Mat<T>::copy<T>, py::const_), "copy")
    //.def("copy", py::overload_cast<Mat<T>&>(&Mat<T>::copy<T>, py::const_), "copy")
    .def("scalarFill", &Mat<T>::scalarFill)
    .def("reshape", py::overload_cast<int>(&Mat<T>::reshape), "change matrix dimensions")
    .def("reshape", py::overload_cast<int, int>(&Mat<T>::reshape), "change matrix dimensions")
    .def("wrap", py::overload_cast<T*, long, size_t*, size_t*>(&Mat<T>::wrap), "use matrix to wrap other container")
    //convenience functions
    .def("zeros", py::overload_cast<size_t>(&Mat<T>::zeros), "returns a 1d matrix of zeros")
    .def("zeros", py::overload_cast<size_t, size_t>(&Mat<T>::zeros), "returns a 2d matrix of zeros")
    .def("zeros_like", &Mat<T>::zeros_like)
    .def("ones", py::overload_cast<size_t>(&Mat<T>::ones), "returns a 1d matrix of ones")
    .def("ones", py::overload_cast<size_t, size_t>(&Mat<T>::ones), "returns a 2d matrix of ones")
    .def("ones_like", &Mat<T>::ones_like)
    .def("empty_like", &Mat<T>::empty_like)
    .def("eye", py::overload_cast<size_t>(&Mat<T>::eye), "returns a 1d identity matrix")
    .def("eye", py::overload_cast<size_t, size_t, int>(&Mat<T>::eye), "returns a 2d identity matrix");
}

PYBIND11_MODULE (Mat_Debug, m){
    declare_debug<bool>(m, "_bool");
    declare_debug<short int>(m, "_short_int");
    declare_debug<unsigned short int>(m, "_unsigned_short_int");
    declare_debug<int>(m, "_int");
    declare_debug<unsigned int>(m, "_unsigned_int");
    declare_debug<long int>(m, "_long_int");
    declare_debug<unsigned long int>(m, "_unsigned_long_int");
    declare_debug<long long int>(m, "_long_long_nt");
    declare_debug<unsigned long long int>(m, "_unsigned_long_long_int");
    declare_debug<signed char>(m, "_signed_char");
    declare_debug<unsigned char>(m, "_unsigned_char");
    declare_debug<char>(m, "_char");
    declare_debug<float>(m, "_float");
    declare_debug<double>(m, "_double");
}