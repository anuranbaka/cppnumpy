#pragma once
#include "../matlib/Mat.h"

template<class Type>
class Mat;
template<class Type>
extern Type minor(const Mat<Type>&, size_t, size_t);

template<class Type>
extern Type determinant(const Mat<Type>&);

template<class Type>
extern Mat<Type> adjugate(const Mat<Type>&);

template<class Type>
extern Mat<Type> inv(const Mat<Type>&);
