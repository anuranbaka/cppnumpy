#include <pybind11/pybind11.h>
#include <Mat.h>
#include <Matmodule.h>

template <class Type>
inline Type TrueDiv(Type a, Type b){ return static_cast<double>(a) / static_cast<double>(b); };
template <class Type>
inline Type FloorDiv(Type a, Type b){ return floor(a / b); };

namespace py = pybind11;

template <class T>
void declare_mat(py::module &m, const std::string &typestr){
    m.def(("buildMat" + typestr).c_str(), []( py::handle obj){
        if(!PyArray_Check(obj.ptr())){
            PyErr_SetString(PyExc_TypeError, "object is not an array");
            return Mat<T>::empty_like(0);
        }
        PyArrayObject* a = reinterpret_cast<PyArrayObject*>(obj.ptr());
        return build_Mat<T>(a);
    });
}

PYBIND11_MODULE (Mat_Pybind, m){
    declare_mat<bool>(m, "_bool");
    declare_mat<short int>(m, "_short_int");
    declare_mat<unsigned short int>(m, "_unsigned_short_int");
    declare_mat<int>(m, "_int");
    declare_mat<unsigned int>(m, "_unsigned_int");
    declare_mat<long int>(m, "_long_int");
    declare_mat<unsigned long int>(m, "_unsigned_long_int");
    declare_mat<long long int>(m, "_long_long_nt");
    declare_mat<unsigned long long int>(m, "_unsigned_long_long_int");
    declare_mat<signed char>(m, "_signed_char");
    declare_mat<unsigned char>(m, "_unsigned_char");
    declare_mat<char>(m, "_char");
    declare_mat<float>(m, "_float");
    declare_mat<double>(m, "_double");
}
