#include <Python.h>
#include <Mat.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

template<class Type = double>
Mat<Type> build_2d_Mat_from_args(Type *data, long int *rows, long int *columns) {
    //first we'll assume a 2d array
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
Mat<Type> build_Mat(PyArrayObject *array){
    if(PyArray_ITEMSIZE(array) != sizeof(Type))
        printf("itemsize mismatch");
    return build_2d_Mat_from_args<Type>((Type*)PyArray_DATA(array), PyArray_DIMS(array), PyArray_DIMS(array)+1);
}
