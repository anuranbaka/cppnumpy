#include "Mat.h"
#include <vector>
using namespace std;

template <class T>
void floodFill(Mat<T>& image, vector<size_t> start, T color, int connectivity = 4){
    floodHelper(image, start, image(start[0],start[1]), color, connectivity); 
}

template <class T>
void floodHelper(Mat<T>& image, vector<size_t> start, T target_color, T new_color, int connectivity = 4){
    if(connectivity != 4 && connectivity != 8){
        fprintf(stderr, "%s", "connectivity must equal 4 or 8");
        exit(1);
    }
    if(image(start[0],start[1]) == target_color){
        image(start[0],start[1]) = new_color;
        for(int i = 0; i < 8; i++){
            if(i == 0 || i == 6 || i == 7) start[1]++;
            if(i == 1) start[0]--;
            if(i == 2 || i == 3) start[1]--;
            if(i == 4 || i == 5) start[0]++;
            if(i%2){
                if(image.inbounds(start[0],start[1]) && connectivity == 8)
                    floodHelper(image, start, target_color, new_color, connectivity);
            }
            else if(image.inbounds(start[0],start[1]))
                floodHelper(image, start, target_color, new_color, connectivity);
        }
    }
    return;
}

template <class T>
void floodFillCustom(Mat<T> image, vector<size_t> start, T color, bool (*fillFunction)(T, T), int connectivity = 4){
    floodCustomHelper(image, start, image(start[0],start[1]), color, fillFunction, connectivity);
}
template <class T>
void floodCustomHelper(Mat<T>& image, vector<size_t> start, T target_color, T new_color, bool (*fillFunction)(T, T), int connectivity = 4){
    if(connectivity != 4 && connectivity != 8){
        fprintf(stderr, "%s", "connectivity must equal 4 or 8");
        exit(1);
    }
    if(fillFunction(image(start[0],start[1]), target_color)){
        image(start[0],start[1]) = new_color;
        for(int i = 0; i < 8; i++){
            if(i == 0 || i == 6 || i == 7) start[1]++;
            if(i == 1) start[0]--;
            if(i == 2 || i == 3) start[1]--;
            if(i == 4 || i == 5) start[0]++;
            if(i%2){
                if(image.inbounds(start[0],start[1]) && connectivity == 8)
                    floodCustomHelper(image, start, target_color, new_color, fillFunction, connectivity);
            }
            else if(image.inbounds(start[0],start[1]))
                floodCustomHelper(image, start, target_color, new_color, fillFunction, connectivity);
        }
    }
    return;
}

