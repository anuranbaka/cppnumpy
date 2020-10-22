#include <pybind11/pybind11.h>
#include <Mat.h>
#include <Matmodule.h>

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
                return false;
            }
            PyArrayObject* a = reinterpret_cast<PyArrayObject*>(src.ptr());
            bool errcheck;
            errcheck = build_Mat(a, value);
            return errcheck && !PyErr_Occurred();
        }

        static handle cast(Mat<T> src, return_value_policy /* policy */, handle /* parent */) {
            src.errorCheck(!src.isContiguous(), "cannot convert non-contiguous matrix");
            int typenum;
            if(std::is_same<T, bool>::value) typenum = NPY_BOOL;
            if(std::is_same<T, short int>::value) typenum = NPY_SHORT;
            if(std::is_same<T, unsigned short int>::value) typenum = NPY_USHORT;
            if(std::is_same<T, int>::value) typenum = NPY_INT;
            if(std::is_same<T, unsigned int>::value) typenum = NPY_UINT;
            if(std::is_same<T, long>::value) typenum = NPY_LONG;
            if(std::is_same<T, unsigned long>::value) typenum = NPY_ULONG;
            if(std::is_same<T, long long>::value) typenum = NPY_LONGLONG;
            if(std::is_same<T, unsigned long long>::value) typenum = NPY_ULONGLONG;
            if(std::is_same<T, signed char>::value) typenum = NPY_BYTE;
            if(std::is_same<T, unsigned char>::value) typenum = NPY_UBYTE;
            if(std::is_same<T, char>::value) typenum = NPY_STRING;//is this right?
            if(std::is_same<T, float>::value) typenum = NPY_FLOAT;
            if(std::is_same<T, double>::value) typenum = NPY_DOUBLE;
            const long int* dims = (long int*)(src.dims);//this could be problematic
            (*src.refCount)++; // this is a memory leak!
            handle result(PyArray_SimpleNewFromData(src.ndims, dims, typenum, src.data));
            return result;
        }
    };
}}