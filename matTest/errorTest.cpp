#include <matMath.h>
#include <Mat.h>
#include <fstream>

double Max(double a, double b){
        if(a > b) return a;
        else return b;
}

int main (){
    Mat<double> valid({1,2,3,4,5,6,7,8},2,4);
    Mat<double> sizeZero(0);
    Mat<double> small(2,1);
    Mat<> output;
    string errorSummary = "";
    printf("errors successfully caught: ");
    try{
        Mat<> errorMat({1.4, 3, -.5},4);
        printf("X");
        errorSummary += "failed to catch invalid initializer\n";
    }
    catch(const exception& e){
        printf("O");
    }
    try{
        output = sizeZero + valid;
        printf("X");
        errorSummary += "failed to catch addition to size 0 mat\n";
    }
    catch(const exception& e){
        printf("O");
    }
    try{
        valid.broadcast(valid, Max, small);
        printf("X");
        errorSummary += "failed to catch storing broadcast result in a small matrix\n";
    }
    catch(const exception& e){
        printf("O");
    }
    try{
        valid.roi(1,2,3,4,5);
        printf("X");
        errorSummary += "failed to catch too many arguments to roi\n";
    }
    catch(const exception& e){
        printf("O");
    }
    try{
        int arr[24];
        size_t newdim[3] = {2,3,4};
        Mat<int>::wrap(arr, -3, newdim);
        printf("X");
        errorSummary += "failed to catch wrapping a container with negative ndim\n";
    }
    catch(const exception& e){
        printf("O");
    }
    try{
        int arr[24];
        size_t newdim[3] = {2,3,4};
        Mat<int>::wrap(arr, 50, newdim);
        printf("X");
        errorSummary += "failed to catch wrapping a container with >32 ndim\n";
    }
    catch(const exception& e){
        printf("O");
    }
    try{
        int arr[24];
        size_t newdim[3] = {2,3,4};
        Mat<int>::wrap(arr, 0, newdim);
        printf("X");
        errorSummary += "failed to catch wrapping a container with 0 ndim\n";
    }
    catch(const exception& e){
        printf("O");
    }
    try{
        valid.reshape(3,-1);
        printf("X");
        errorSummary += "failed to catch invalid reshape dimension inference\n";
    }
    catch(const exception& e){
        printf("O");
    }
    try{
        valid.reshape(-1,-1);
        printf("X");
        errorSummary += "failed to catch multiple inferred dimensions in reshape\n";
    }
    catch(const exception& e){
        printf("O");
    }
    try{
        valid.reshape(-2,4);
        printf("X");
        errorSummary += "failed to catch negative dimension in reshape\n";
    }
    catch(const exception& e){
        printf("O");
    }
    try{
        Mat<double> similar({1,2,3,4,5,6,7,8},2,4);
        for(Mat<double>::iterator i = valid.begin(); i != similar.end(); ++i){
            *i = 1;
        }
        printf("X");
        errorSummary += "failed to catch comparison between iterators of different matrices\n";
    }
    catch(const exception& e){
        printf("O");
    }
    try{
        Mat<size_t>::arange(3,7,0);
        printf("X");
        errorSummary += "failed to catch arange with step = 0\n";
    }
    catch(const exception& e){
        printf("O");
    }

    if(errorSummary == ""){
        printf("\nAll errors caught!\n");
    }
    else
    {
        printf("\nError Summary:\n%s", errorSummary.c_str());
    }
    
    return 0;
}