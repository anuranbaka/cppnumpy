#include "../matlib/Mat.h"
#include <fstream>

int main (){
    Mat<double> m({1,2,3,4,5,6,7,8},2,4);
    Mat<> n({1,2,3},1,3);
    Mat<> x({1,2,3,4,5,6,7,8,9},3,3);
    Mat<> y({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24},4,6);
    Mat<> output(1,1);
    FILE* outFile;
    outFile = fopen("Mat_Test/Mat_test_output.txt", "w");

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

    output = m.roi(1,3);
    fprintf(outFile, "m.roi(1,3)\n");
    output.print(outFile);

    output = -x.roi(1,-1,1);
    fprintf(outFile, "-x.roi(1,-1,1)\n");
    output.print(outFile);

    y.roi(2,5,1,3).scalarFill(0);
    fprintf(outFile, "y.roi(2,5,1,3).scalarFill(0)\n");
    y.print(outFile);

    return 0;
}