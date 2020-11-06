#include <pybind11/pybind11.h>
#include <Mat.h>
#include <pythonAPI.h>
namespace py = pybind11;

#ifndef _MSC_VER
#include <cxxabi.h>
#define DEMANGLE_TYPEID_NAME(x) abi::__cxa_demangle(typeid(x).name(), NULL, NULL, NULL)
#else
#define DEMANGLE_TYPEID_NAME(x) typeid((x)).name()
#endif

int init_numpy(){
     import_array1(0); // PyError if not successful
     return 1;
}

const static int numpy_initialized =  init_numpy();

namespace pybind11 { namespace detail {
    template <class T>
    struct type_caster<Mat<T>> {
    public:
        PYBIND11_TYPE_CASTER(Mat<T>, _("Mat<T>"));

        bool load(handle src, bool) {
            if(!PyArray_Check(src.ptr())){
                PyErr_SetString(PyExc_TypeError, "source handle is not an array");
                throw py::error_already_set();
                return PyErr_Occurred();
            }
            if(getTypenum<T>() == -1){
                static char* tName = []{static char tName[255];
                snprintf(tName, 255, "Conversion to C++ Mat type %s unsupported",
                        DEMANGLE_TYPEID_NAME(T));return tName;}();
                PyErr_SetString(PyExc_TypeError, tName);
                throw py::error_already_set();
                return PyErr_Occurred();
            }
            if(!PyArray_EquivTypenums(getTypenum<T>(), PyArray_TYPE(arr))){
                return PyErr_Occurred();
            }
            PyArrayObject* arr = (PyArrayObject*) src.ptr();
            value = wrap_numpy<T>(src.ptr());
            return !PyErr_Occurred();
        }

        static handle cast(Mat<T> src, return_value_policy /* policy */, handle /* parent */) {
            handle result(wrap_mat<T>(src));
            return result;
        }
    };
}}