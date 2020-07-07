#include "../matlib/Mat.h"
#include <fstream>

double Max(double a, double b){
        if(a > b) return a;
        else return b;
};

int main (){
    Mat<double> m({1,2,3,4,5,6,7,8},2,4);
    Mat<> n({1,2,3},1,3);
    Mat<> x({1,2,3,4,5,6,7,8,9},3,3);
    Mat<> y({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24},4,6);
    Mat<> output(1,1);
    FILE* outFile = fopen("projects/Mat_Test/Mat_test_output.txt", "w");

    fprintf(outFile, "matrix m\n");
    m.print(outFile);

    fprintf(outFile, "matrix n\n");
    n.print(outFile);

    fprintf(outFile, "matrix x\n");
    x.print(outFile);

    fprintf(outFile, "matrix y\n");
    y.print(outFile);

    output = x.t();
    fprintf(outFile, "transpose of x\n");
    output.print(outFile);

    output = m.t();
    fprintf(outFile, "transpose of m\n");
    output.print(outFile);

    output = n + x;
    fprintf(outFile, "n + x\n");
    output.print(outFile);

    output = n.t() + x;
    fprintf(outFile, "transpose of n + x\n");
    output.print(outFile);

    output = x.t() - n.t();
    fprintf(outFile, "transpose of x - transpose of n\n");
    output.print(outFile);

    output = n ^ x;
    fprintf(outFile, "matrix multiplication n ^ x\n");
    output.print(outFile);

    Mat<> newMat(2,4);
    m.copy(newMat);
    fprintf(outFile, "m.copy(newMat)\n");
    newMat.print(outFile);

    output = m.roi(0,1,1,3);
    fprintf(outFile, "m.roi(0,1,1,3)\n");
    output.print(outFile);

    output = -x.roi(1,-1,1);
    fprintf(outFile, "-x.roi(1,-1,1)\n");
    output.print(outFile);

    y.roi(1,3,2,5).scalarFill(0);
    fprintf(outFile, "y.roi(1,3,2,5).scalarFill(0)\n");
    y.print(outFile);

    Mat<> smallMat ({5},1,1);
    fprintf(outFile, "Printing a 1x1 matrix\n");
    smallMat.print(outFile);

    Mat<> smallerMat (0,0);
    fprintf(outFile, "Printing a 0x0 matrix\n");
    smallerMat.print(outFile);
    fprintf(outFile, "(should be nothing above this line)\n");

    Mat<> justFours ({4,4,4}, 1, 3);
    output = x.broadcast(justFours, Max);
    fprintf(outFile, "Set each element of x to a minimum of 4 with broadcast()\n");
    output.print(outFile);

    Mat<> a({1,2,3,4,5,6}, 6);
    Mat<> b({3,3,3,3,3,3}, 6);
    Mat<> c({5}, 1);
    Mat<> d(0);

    fprintf(outFile, "1d matrix a\n");
    a.print(outFile);
    fprintf(outFile, "1d matrix b\n");
    b.print(outFile);
    fprintf(outFile, "1d matrix c\n");
    c.print(outFile);
    fprintf(outFile, "empty 1d matrix d\n");
    fprintf(outFile, "(should be nothing above this line)\n");
    d.print(outFile);

    output = a + b;
    fprintf(outFile, "a + b (assignment to previously 2d matrix)\n");
    output.print(outFile);

    output = c - a;
    fprintf(outFile, "c - a\n");
    output.print(outFile);

    output = -a;
    fprintf(outFile, "-a\n");
    output.print(outFile);

    output = a.roi(2,5);
    fprintf(outFile, "matrix a, elements 2 to 4\n");
    output.print(outFile);

    output = 7;
    fprintf(outFile, "assignment to 7s\n");
    output.print(outFile);

    output = Mat<>::zeros(8,8);
    fprintf(outFile, "8x8 matrix of zeros\n");
    output.print(outFile);
/*
    output = Mat<>::ones(4,5);
    fprintf(outFile, "4x5 matrix of ones\n");
    output.print(outFile);
    

    output = Mat<>::identity(6);
    fprintf(outFile, "6x6 identity matrix\n");
    output.print(outFile);
*/
    return 0;
}