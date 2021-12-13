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
void wrap_numpy_allocateMeta(Mat<T> &new_mat, void*, const long new_ndim){
    new_mat.dims = (size_t*)PyMem_Malloc(new_ndim*sizeof(size_t));
    new_mat.strides = (size_t*)PyMem_Malloc(new_ndim*sizeof(size_t));
    if(new_mat.base == NULL){
        MatBase<T>* newBase;
        newBase = (MatBase<T>*)PyMem_Malloc(sizeof(MatBase<T>));
        new_mat.base = newBase;
    }
}

template <class T>
void wrap_numpy_deallocateMeta(Mat<T> &mat){
    PyMem_Free(mat.dims);
    PyMem_Free(mat.strides);
    if(mat.base->refCount <= 0) PyMem_Free(mat.base);
}

template <class T>
void wrap_numpy_allocateData(MatBase<T> &base, void*, const size_t size){
    const long temp = size;
    const npy_intp* dim = &temp;
    base.data = (T*)PyArray_DATA(reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dim, getTypenum<T>())));
}

template <class T>
void wrap_numpy_deallocateData(MatBase<T> &base){
    Py_DECREF(base.allocator->userdata);
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
    npy_intp py_strides[MAX_NDIM];
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

    static AllocInfo<T> npInfo;
    npInfo.userdata = arrBase;
    npInfo.allocateMeta = *wrap_numpy_allocateMeta<T>;
    npInfo.deallocateMeta = *wrap_numpy_deallocateMeta<T>;
    npInfo.allocateData = *wrap_numpy_allocateData<T>;
    npInfo.deallocateData = *wrap_numpy_deallocateData<T>;

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
