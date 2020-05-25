#include "Mat.h"

int main (){
    Mat<double> m({1,2,3,4,5,6,7,8},2,4);
    Mat<> n({1,2,3},1,3);
    Mat<> x({1,2,3,4,5,6,7,8,9},3,3);

    printf("matrix m\n");
    m.print();

    printf("matrix n\n");
    n.print();

    printf("matrix x\n");
    x.print();

    x = m;
    printf("matrix x\n");
    x.print();

    x = n;
    printf("matrix x\n");
    x.print();

    return 0;
}