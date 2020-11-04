#include "../include/matMath.h"

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
template <class Type>
Mat<Type> inv(const Mat<Type>& mat){
    mat.errorCheck(mat.ndims != 2 || mat.rows() != mat.columns(),
        "Matrix dimensions non-invertible");
    Type det = determinant(mat);
    mat.errorCheck(det == 0, "matrix is singular");
    return adjugate(mat)/det;
}
template Mat<double> inv<double>(const Mat<double>& mat);
template Mat<float> inv<float>(const Mat<float>& mat);