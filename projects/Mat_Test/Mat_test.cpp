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

    output = m.roi(0,2,1,3);
    fprintf(outFile, "m.roi(0,1,1,3)\n");
    output.print(outFile);

    output = -x.roi(1,-1,1);
    fprintf(outFile, "-x.roi(1,-1,1)\n");
    output.print(outFile);

    output = y.roi(1,3);
    fprintf(outFile, "y.roi(1,3)\n");
    output.print(outFile);

    output = y.roi(1);
    fprintf(outFile, "y.roi(1)\n");
    output.print(outFile);

    output = x.roi();
    fprintf(outFile, "x.roi()\n");
    output.print(outFile);

    fprintf(outFile, "m.T()\n");
    m.T().print(outFile);

    output = Mat<double>::empty_like(m);
    output.reshape(m.columns(),m.rows());
    m.T(output);
    fprintf(outFile, "m.T(output)\n");
    output.print(outFile);

    x.T();
    fprintf(outFile, "x.T()\n");
    x.print(outFile);
    x.T();

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

    output = x + 5;
    fprintf(outFile, "broadcast scalar: x + 5\n");
    output.print(outFile);

    output = 2520.0 / x;
    fprintf(outFile, "broadcast scalar: 2520 / x\n");
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

    output = Mat<>::ones(4,5);
    fprintf(outFile, "4x5 matrix of ones\n");
    output.print(outFile);
    
    output = Mat<>::eye(6);
    fprintf(outFile, "6x6 identity matrix\n");
    output.print(outFile);

    output = Mat<>::eye(4,7,1);
    fprintf(outFile, "4x7 identity matrix with diagonal at 1\n");
    output.print(outFile);

    output = Mat<>::eye(4,7,-2);
    fprintf(outFile, "4x7 identity matrix with diagonal at -2\n");
    output.print(outFile);

    output.reshape(14,2);
    fprintf(outFile, "reshape the previous matrix into a 14x2 matrix\n");
    output.print(outFile);

    output = x;
    output.reshape(9);
    fprintf(outFile, "matrix x as a 1d, 9 element matrix\n");
    output.print(outFile);

    output = y.roi(1,3);
    output.reshape(6,2);
    fprintf(outFile, "taking the two middle rows from y and reshaping to a 6x2 matrix\n");
    fprintf(outFile, "(legal because full rows are taken and are contiguous)\n");
    fprintf(outFile, "(Note that the zeros we filled in before are still there)\n");
    output.print(outFile);

    double e[6] = {1,2,3,4,5,6};
    size_t shape[2] = {2,3};
    int64_t counter = 1;
    Mat<double> wrapper = Mat<double>::wrap(6, e, 2, shape);
    fprintf(outFile, "6 element c-style array wrapped into a 2x3 matrix\n");
    wrapper.print(outFile);
    
    wrapper = Mat<double>::wrap(6, e, 2, shape, &counter);
    fprintf(outFile, "6 element c-style array wrapped into a 2x3 matrix with preset refcount\n");
    wrapper.print(outFile);

    Mat<bool> boolMat({true, true, false},1,3);
    Mat<bool> outBool;
    fprintf(outFile, "boolMat:\n");

    boolMat.print(outFile);
    outBool = boolMat & x;
    fprintf(outFile, "boolMat & matrix x\n");
    outBool.print(outFile);

    outBool = x & !boolMat;
    fprintf(outFile, "matrix x & !boolMat\n");
    outBool.print(outFile);

    outBool = boolMat | !x;
    fprintf(outFile, "boolMat | !x\n");
    outBool.print(outFile);

    outBool = x | boolMat;
    fprintf(outFile, "x | boolMat\n");
    outBool.print(outFile);

    outBool = true & y;
    fprintf(outFile, "true & y\n");
    outBool.print(outFile);

    outBool = !y | false;
    fprintf(outFile, "!y | false\n");
    outBool.print(outFile);

    outBool = !boolMat & true;
    fprintf(outFile, "!boolMat | true\n");
    outBool.print(outFile);

    return 0;
}