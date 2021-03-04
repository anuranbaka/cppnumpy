#include "../include/matMath.h"
extern "C"{
    #include <clapack.h>
}

template<class Type>
Type determinant(const Mat<Type>& mat){
    if(mat.columns() == 1){
        return mat(0,0);
    }
    if(mat.columns() == 2){
        return mat(0,0)*mat(1,1) - mat(0,1)*mat(1,0);
    }
    else{
        Type result = 0;
        for(int i = 0; i < mat.columns(); i++){
            if(i%2 == 0) result += mat(0,i)*minor(mat,0,i);
            else result -= mat(0,i)*minor(mat,0,i);
        }
        return result;
    }
}
template<class Type>
Mat<Type> adjugate(const Mat<Type>& mat){
    Mat<Type> result(Mat<Type>::empty_like(mat));
    for(int i = 0; i < result.rows(); i++){
        for(int j = 0; j < result.columns(); j++){
            if(i%2 ^ j%2) result(i,j) = -1*minor(mat, i, j);
            else result(i,j) = minor(mat, i, j);
        }
    }
    return result.T();
}

template<class Type>
Type minor(const Mat<Type>& mat, size_t x, size_t y){
    Mat<Type> temp(mat.rows()-1,mat.columns()-1);
    size_t inRow = 0, inCol = 0;
    for(int i = 0; i < temp.rows(); i++){
        if(i == x) inRow++;
        inCol = 0;
        for(int j = 0; j < temp.columns(); j++){
            if(j == y) inCol++;
            temp(i,j) = mat(inRow,inCol);
            inCol++;
        }
        inRow++;
    }
    return determinant(temp);
}
template <>
Mat<double> inv(const Mat<double>& mat){
    if(mat.rows() != mat.columns())
        throw invalid_argument("inverse can only be called on a square matrix");
    if(mat.ndim != 2)
        throw invalid_argument("inverse can only be called on a 2d matrix");
    if(!mat.isContiguous())
        throw invalid_argument("inverse can only be called on a contiguous matrix");
    Mat<double> result(mat.copy());
    int order = mat.columns();
    int* piv;
    if(order < 1000){
        piv = (int*)alloca(order*sizeof(int));
    }
    else{
        piv = (int*)malloc(order*sizeof(int));
    }
    clapack_dgetrf(CblasRowMajor, order, order, result.data, order, piv);
    if(clapack_dgetri(CblasRowMajor, order, result.data, order, piv) != 0)
        throw runtime_error("Lapack inverse failed");
    if(order >= 1000){
        free(piv);
    }
    return result;
}
template <>
Mat<float> inv<float>(const Mat<float>& mat){
    if(mat.rows() != mat.columns())
        throw invalid_argument("inverse can only be called on a square matrix");
    if(mat.ndim != 2)
        throw invalid_argument("inverse can only be called on a 2d matrix");
    if(!mat.isContiguous())
        throw invalid_argument("inverse can only be called on a contiguous matrix");
    Mat<float> result(mat.copy());
    int order = mat.columns();
    int* piv;
    if(order < 1000){
        piv = (int*)alloca(order*sizeof(int));
    }
    else{
        piv = (int*)malloc(order*sizeof(int));
    }
    clapack_sgetrf(CblasRowMajor, order, order, result.data, order, piv);
    if(clapack_sgetri(CblasRowMajor, order, result.data, order, piv) != 0)
        throw invalid_argument("Lapack inverse failed");
    if(order >= 1000){
        free(piv);
    }
    return result;
};
template Mat<double> inv<double>(const Mat<double>& mat);
template Mat<float> inv<float>(const Mat<float>& mat);