#include <Python.h>
#include <Mat.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

static const char* MAT_NAME = "Mat";

template<class T>
int getTypenum(){
    if(std::is_same<T, bool>::value) return NPY_BOOL;
    if(std::is_same<T, signed char>::value) return NPY_BYTE;
    if(std::is_same<T, unsigned char>::value) return NPY_UBYTE;
    if(std::is_same<T, int16_t>::value) return NPY_SHORT;
    if(std::is_same<T, uint16_t>::value) return NPY_USHORT;
    if(std::is_same<T, int32_t>::value) return NPY_INT;
    if(std::is_same<T, uint32_t>::value) return NPY_UINT;
    if(std::is_same<T, long long>::value ||
        std::is_same<T, int64_t>::value) return NPY_LONGLONG;
    if(std::is_same<T, unsigned long long>::value ||
        std::is_same<T, uint64_t>::value) return NPY_ULONGLONG;
    if(std::is_same<T, float>::value) return NPY_FLOAT;
    if(std::is_same<T, double>::value) return NPY_DOUBLE;
    if(std::is_same<T, char>::value){
        T test = 0;
        if(test - (T)1 > 0) return NPY_UBYTE;
        else return NPY_BYTE;
    }
    return -1;
}

template <class T>
void wrap_numpy_allocate(Mat<T>* new_mat, void* userdata, const long new_ndim){
    new_mat->dims = new size_t[new_ndim];
    new_mat->strides = new size_t[new_ndim];
    new_mat->allocator->userdata = userdata;
    Py_INCREF((PyObject*)userdata);
}

template <class T>
void wrap_numpy_deallocate(Mat<T>* mat){
    delete[] mat->dims;
    delete[] mat->strides;
    Py_DECREF((PyObject*)mat->allocator->userdata);
}

template <class T>
static void destructor_wrapper(PyObject* o){
    void* ptr = PyCapsule_GetPointer(o, MAT_NAME);
    Mat<T>* m = (Mat<T>*)ptr;
    delete m;
}

template<class T>
PyObject* wrap_mat(Mat<T>& cmat){
    Mat<T>* temp = new Mat<T>(cmat);
    PyObject* capsule = PyCapsule_New(temp, MAT_NAME, destructor_wrapper<T>);
    int writeFlag = 0;
    if(!is_const<T>()) writeFlag = NPY_ARRAY_WRITEABLE;
    npy_intp py_strides[cmat.ndim];
    for(long i = 0; i < cmat.ndim; i++){
        py_strides[i] = cmat.strides[i]*sizeof(T);
    }
    PyObject* out = PyArray_New(&PyArray_Type,
                                    cmat.ndim,
                                    (npy_intp*)cmat.dims,
                                    getTypenum<T>(),
                                    py_strides,
                                    cmat.data,
                                    (int)sizeof(T),
                                    writeFlag, NULL);
    PyArray_UpdateFlags(reinterpret_cast<PyArrayObject*>(out),
                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(out), capsule);
    return out;
}

template<class T>
Mat<T> wrap_numpy(PyArrayObject* arr){
    PyObject* arrBase = (PyObject*)arr;
    long nd = PyArray_NDIM(arr);
    size_t* mat_strides = new size_t[nd];
    Mat<T> out;
    for(long i = 0; i < nd; i++){
        if((PyArray_STRIDES(arr)[i])%PyArray_ITEMSIZE(arr) != 0)
            throw invalid_argument("Strides must be a multiple of ITEMSIZE to convert to Mat");
        mat_strides[i] = (PyArray_STRIDES(arr)[i])/PyArray_ITEMSIZE(arr);
    }

    AllocInfo<T> npInfo;
    npInfo.userdata = arrBase;
    npInfo.allocateMeta = *wrap_numpy_allocate<T>;
    npInfo.deallocateMeta = *wrap_numpy_deallocate<T>;

    out = Mat<T>::wrap(
        (T*)PyArray_DATA(arr),
        PyArray_NDIM(arr),
        reinterpret_cast<typename Mat<T>::size_type*>(PyArray_DIMS(arr)),
        mat_strides,
        &npInfo);
    delete[] mat_strides;
    Py_INCREF(arrBase);
    return out;
}
