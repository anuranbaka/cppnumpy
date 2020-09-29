#include <pybind11/pybind11.h>
#include <Mat.h>
#include "../Numpy/Matmodule.h"

namespace py = pybind11;

template <class Type>
inline Type TrueDiv(Type a, Type b){ return static_cast<double>(a) / static_cast<double>(b); };
template <class Type>
inline Type FloorDiv(Type a, Type b){ return floor(a / b); };

PYBIND11_MODULE (Mat_Pybind, m){
    py::class_<Mat<double>>(m, "Mat")
        .def(py::init<size_t>())
        .def(py::init<size_t, size_t>())
        .def(py::init<std::initializer_list<double>, size_t>())
        .def(py::init<std::initializer_list<double>, size_t, size_t>())
        //iterator/capacity functions
        .def("size", &Mat<double>::size)
        .def("rows", &Mat<double>::rows)
        .def("columns", &Mat<double>::columns)
        .def("inbounds", py::overload_cast<size_t>(&Mat<double>::inbounds), "checks if given coordinates are inbounds")
        //element access/assignment
        .def("__call__", py::overload_cast<size_t>(&Mat<double>::operator()))
        .def("__call__", py::overload_cast<size_t, size_t>(&Mat<double>::operator()))
        .def("assign", py::overload_cast<const Mat<double>&>(&Mat<double>::operator=))
        .def("assign", py::overload_cast<double>(&Mat<double>::operator=))
        //arithmetic operations
        .def("__add__", py::overload_cast<const Mat<double>&>(&Mat<double>::operator+))
        .def("__add__", py::overload_cast<double>(&Mat<double>::operator+))
        .def("__radd__", py::overload_cast<double>(&Mat<double>::operator+))
        .def("__iadd__", py::overload_cast<const Mat<double>&>(&Mat<double>::operator+=))
        .def("__iadd__", py::overload_cast<double>(&Mat<double>::operator+=))
        .def("__sub__", py::overload_cast<const Mat<double>&>(&Mat<double>::operator-))
        .def("__sub__", py::overload_cast<double>(&Mat<double>::operator-))
        .def("__rsub__", [](Mat<double> &a, double b) {
            return b - a;
        }, py::is_operator())
        .def("__isub__", py::overload_cast<const Mat<double>&>(&Mat<double>::operator-=))
        .def("__isub__", py::overload_cast<double>(&Mat<double>::operator-=))
        .def("__mul__", py::overload_cast<const Mat<double>&>(&Mat<double>::operator*))
        .def("__mul__", py::overload_cast<double>(&Mat<double>::operator*))
        .def("__rmul__", py::overload_cast<double>(&Mat<double>::operator*))
        .def("__imul__", py::overload_cast<const Mat<double>&>(&Mat<double>::operator*=))
        .def("__imul__", py::overload_cast<double>(&Mat<double>::operator*=))
        .def("__truediv__", py::overload_cast<const Mat<double>&>(&Mat<double>::operator/))
        .def("__truediv__", py::overload_cast<double>(&Mat<double>::operator/))
        .def("__rtruediv__", [](Mat<double> &a, double b) {
            return b / a;
        }, py::is_operator())
        .def("__itruediv__", py::overload_cast<const Mat<double>&>(&Mat<double>::operator/=))
        .def("__itruediv__", py::overload_cast<double>(&Mat<double>::operator/=))
        .def("__neg__", py::overload_cast<const Mat<double>&>(&Mat<double>::operator-))
        .def("__matmul__", &Mat<double>::operator^)
        .def("T", py::overload_cast<>(&Mat<double>::T), "hard transpose")
        .def("t", &Mat<double>::t)
        //logical operations
        .def("__and__", py::overload_cast<const Mat<double>&>(&Mat<double>::operator&<double>))
        .def("__and__", (Mat<bool> (Mat<double>::*)(bool))(&Mat<double>::operator&))
        .def("__rand__", (Mat<bool> (Mat<double>::*)(bool))(&Mat<double>::operator&))
        .def("__or__", py::overload_cast<const Mat<double>&>(&Mat<double>::operator|<double>))
        .def("__or__", (Mat<bool> (Mat<double>::*)(bool))(&Mat<double>::operator|))
        .def("__ror__", (Mat<bool> (Mat<double>::*)(bool))(&Mat<double>::operator|))
        .def("not", &Mat<double>::operator!)
        .def("__eq__", py::overload_cast<double>(&Mat<double>::operator==))
        .def("__eq__", py::overload_cast<const Mat<double>>(&Mat<double>::operator==))
        .def("__eq__", py::overload_cast<double>(&Mat<double>::operator==))
        .def("__ne__", py::overload_cast<const Mat<double>>(&Mat<double>::operator!=))
        .def("__ne__", py::overload_cast<double>(&Mat<double>::operator!=))
        .def("__lt__", py::overload_cast<const Mat<double>>(&Mat<double>::operator<))
        .def("__lt__", py::overload_cast<double>(&Mat<double>::operator<))
        .def("__le__", py::overload_cast<const Mat<double>>(&Mat<double>::operator<=))
        .def("__le__", py::overload_cast<double>(&Mat<double>::operator<=))
        .def("__gt__", py::overload_cast<const Mat<double>>(&Mat<double>::operator>))
        .def("__gt__", py::overload_cast<double>(&Mat<double>::operator>))
        .def("__ge__", py::overload_cast<const Mat<double>>(&Mat<double>::operator>=))
        .def("__ge__", py::overload_cast<double>(&Mat<double>::operator>=))
        .def("all", &Mat<double>::all)
        .def("any", &Mat<double>::any)
        //meta functions
        .def("broadcast", py::overload_cast<const Mat<double>&, double (*)(double, double)>(&Mat<double>::broadcast<double, double>), "elementwise broadcast function")
        .def("broadcast", py::overload_cast<double, double (*)(double, double)>(&Mat<double>::broadcast<double, double>), "elementwise broadcast function")
        .def("broadcast", py::overload_cast<double, Mat<double>&, double (*)(double, double)>(&Mat<double>::broadcast<double, double>), "elementwise broadcast function")
        .def("roi", &Mat<double>::roi)
        .def("print", py::overload_cast<>(&Mat<double>::print), "print contents of matrix")
        .def("copy", py::overload_cast<>(&Mat<double>::copy<double>, py::const_), "copy")
        .def("copy", py::overload_cast<Mat<double>&>(&Mat<double>::copy<double>, py::const_), "copy")
        .def("scalarFill", &Mat<double>::scalarFill)
        .def("reshape", py::overload_cast<int>(&Mat<double>::reshape), "change matrix dimensions")
        .def("reshape", py::overload_cast<int, int>(&Mat<double>::reshape), "change matrix dimensions")
        .def("wrap", py::overload_cast<size_t, double*, size_t, size_t*>(&Mat<double>::wrap), "use matrix to wrap other container")
        .def("wrap", py::overload_cast<size_t, double*, size_t, size_t*, int64_t*>(&Mat<double>::wrap), "use matrix to wrap other container")
        //convenience functions
        .def("zeros", py::overload_cast<size_t>(&Mat<double>::zeros), "returns a 1d matrix of zeros")
        .def("zeros", py::overload_cast<size_t, size_t>(&Mat<double>::zeros), "returns a 2d matrix of zeros")
        .def("zeros_like", &Mat<double>::zeros_like)
        .def("ones", py::overload_cast<size_t>(&Mat<double>::ones), "returns a 1d matrix of ones")
        .def("ones", py::overload_cast<size_t, size_t>(&Mat<double>::ones), "returns a 2d matrix of ones")
        .def("ones_like", &Mat<double>::ones_like)
        .def("empty_like", &Mat<double>::empty_like)
        .def("eye", py::overload_cast<size_t>(&Mat<double>::eye), "returns a 1d identity matrix")
        .def("eye", py::overload_cast<size_t, size_t, int>(&Mat<double>::eye), "returns a 2d identity matrix");
    m.def("buildMat", []( py::handle obj){
        //check that it's an array
        PyArrayObject* a = reinterpret_cast<PyArrayObject*>(obj.ptr());
        return build_Mat<double>(a);
    });
}