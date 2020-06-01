#include "Mat.h"

int main (){
    Mat<double> m({1,2,3,4,5,6,7,8},2,4);
    Mat<> n({1,2,3},1,3);
    Mat<> x({1,2,3,4,5,6,7,8,9},3,3);
    Mat<> output(1,1);

    printf("matrix m\n");
    m.print();

    printf("matrix n\n");
    n.print();

    printf("matrix x\n");
    x.print();

    output = x.t();
    printf("transpose of x\n");
    output.print();

    output = m.t();
    printf("transpose of m\n");
    output.print();

    output = n.t();
    printf("transpose of n\n");
    output.print();

    output = n + x;
    printf("n + x\n");
    output.print();

    output = n.t() + x;
    printf("transpose of n + x\n");
    output.print();

    output = x.t() - n.t();
    printf("transpose of x - transpose of n\n");
    output.print();

    output = n ^ x;
    printf("matrix multiplication n ^ x\n");
    output.print();

    Mat<> newMat(2,4);
    m.copy(newMat);
    printf("newMat = m\n");
    newMat.print();

    return 0;
}