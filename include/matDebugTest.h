#pragma once
#include <Mat.h>
template <class T>
Mat<T> copyTest(Mat<T> image){
    return image.copy();
}