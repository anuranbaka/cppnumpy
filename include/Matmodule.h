#include <Python.h>
#include <Mat.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

template<class Type>
Mat<Type> build_2d_Mat_from_args(Type *data, long int *rows, long int *columns) {
    //arguments should be a (ndarray.data, ndarray.shape[0], ndarray.shape[1])

    Mat<Type> result(*rows, *columns);
    for(int i = 0; i < *rows; i++){
        for(int j = 0; j < *columns; j++){
            result(i,j) = data[(i*(*columns))+j];
        }
    }

    return result;
}

template<class Type>
Mat<Type> build_1d_Mat_from_args(Type *data, long int *columns) {
    //arguments should be a (ndarray.data, ndarray.shape[0])
    Mat<Type> result(*columns);
    for(int j = 0; j < *columns; j++){
        result(j) = data[j];
    }
    return result;
}

template<class Type>
bool build_Mat(PyArrayObject *array, Mat<Type>& new_mat){
    if(PyArray_ITEMSIZE(array) != sizeof(Type)){
        printf("itemsize mismatch\n");
        fflush(stdout);
        return false;
    }
    if(PyArray_NDIM(array) == 1){
        new_mat = build_1d_Mat_from_args<Type>((Type*)PyArray_DATA(array), PyArray_DIMS(array));
        return true;
    }
    else if(PyArray_NDIM(array) == 2){
        new_mat = build_2d_Mat_from_args<Type>((Type*)PyArray_DATA(array), PyArray_DIMS(array), PyArray_DIMS(array)+1);
        return true;
    }
    else{
        printf("n-dimensional array not yet supported\n");
        fflush(stdout);
        return false;
    }
}
