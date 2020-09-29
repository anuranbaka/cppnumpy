#pragma once
#include "../include/Mat.h"
#include <vector>

template <class T>
void floodFill(Mat<T>& image, vector<size_t> start, T color, int connectivity = 4);

template <class T>
void floodHelper(Mat<T>& image, vector<size_t> start, T target_color, T new_color, int connectivity = 4);