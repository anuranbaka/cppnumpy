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

template<class T>
Mat<T> build_2d_Mat_from_args(T *data, long int *rows, long int *columns) {
    //arguments should be a (ndarray.data, ndarray.shape[0], ndarray.shape[1])

    Mat<T> result(*rows, *columns);
    for(int i = 0; i < *rows; i++){
        for(int j = 0; j < *columns; j++){
            result(i,j) = data[(i*(*columns))+j];
        }
    }

    return result;
}

template<class T>
Mat<T> build_1d_Mat_from_args(T *data, long int *columns) {
    //arguments should be a (ndarray.data, ndarray.shape[0])
    Mat<T> result(*columns);
    for(int j = 0; j < *columns; j++){
        result(j) = data[j];
    }
    return result;
}

template<class T>
bool build_Mat(PyArrayObject *array, Mat<T>& new_mat){
    if(PyArray_ITEMSIZE(array) != sizeof(T)){
        printf("itemsize mismatch\n");
        fflush(stdout);
        return false;
    }
    if(PyArray_NDIM(array) == 1){
        new_mat = build_1d_Mat_from_args<T>((T*)PyArray_DATA(array),
                                            PyArray_DIMS(array));
        return true;
    }
    else if(PyArray_NDIM(array) == 2){
        new_mat = build_2d_Mat_from_args<T>((T*)PyArray_DATA(array),
                                            PyArray_DIMS(array), PyArray_DIMS(array)+1);
        return true;
    }
    else{
        printf("n-dimensional array not yet supported\n");
        fflush(stdout);
        return false;
    }
}

template <class T>
void destructor_wrapper(PyObject* o){
    void* ptr = PyCapsule_GetPointer(o, MAT_NAME);
    Mat<T>* m = (Mat<T>*)ptr;
    delete m;
}

template<class T>
void NumpyDestructor(Mat<T>* mat, void* data){
    Py_DECREF((PyObject*)data);
}

template<class T>
PyObject* wrap_mat(Mat<T>& cmat){
    Mat<T>* temp = new Mat<T>(cmat);
    PyObject* capsule = PyCapsule_New(temp, MAT_NAME, destructor_wrapper<T>);
    int writeFlag = 0;
    if(!is_const<T>()) writeFlag = NPY_ARRAY_WRITEABLE;
    npy_intp* py_strides = new npy_intp[cmat.ndims];
    for(long i = 0; i < cmat.ndims; i++){
        py_strides[i] = cmat.strides[i]*sizeof(T);
    }
    PyObject* out = PyArray_New(&PyArray_Type,
                                    cmat.ndims,
                                    (npy_intp*)cmat.dims,
                                    getTypenum<T>(),
                                    py_strides,
                                    cmat.data,
                                    (int)sizeof(T),
                                    writeFlag, NULL);
    PyArray_UpdateFlags(reinterpret_cast<PyArrayObject*>(out),
                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(out), capsule);
    delete[] py_strides;
    return out;
}

template<class T>
Mat<T> wrap_numpy(PyObject* arrIn){
    if(!PyArray_Check(arrIn)){
        PyErr_SetString(PyExc_TypeError, "source handle is not an array");
        return Mat<T>::zeros(0);
    }
    PyArrayObject* arr = (PyArrayObject*) arrIn;
    long nd = PyArray_NDIM(arr);
    size_t* mat_strides = new size_t[nd];
    for(long i = 0; i < nd; i++){
        mat_strides[i] = (PyArray_STRIDES(arr)[i])/PyArray_ITEMSIZE(arr);
    }
    Mat<T> out = Mat<T>::wrap(
        (T*)PyArray_DATA(arr),
        PyArray_NDIM(arr),
        reinterpret_cast<typename Mat<T>::size_type*>(PyArray_DIMS(arr)),
        mat_strides,
        &Py_REFCNT(arrIn),
        NumpyDestructor<T>,
        arrIn);
    delete[] mat_strides;
    Py_INCREF(arrIn);
    return out;
}
