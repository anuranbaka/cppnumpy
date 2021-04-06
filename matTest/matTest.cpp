#include <matMath.h>
#include <Mat.h>
#include <fstream>

double Max(double a, double b){
        if(a > b) return a;
        else return b;
}

int main (){
    Mat<double> m({1,2,3,4,5,6,7,8},2,4);
    Mat<> n({1,2,3},1,3);
    Mat<> o({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},2,2,4);
    Mat<> x({1,2,3,4,5,6,7,8,9},3,3);
    Mat<> y({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24},4,6);
    Mat<> z(4,1,4,6);
    { int i = 1; for(auto& j : z){ j = i++; }}
    Mat<> output;
    FILE* outFile = fopen("matTest/matTestOutput.txt", "w");

    fprintf(outFile, "matrix m\n");
    m.print(outFile);

    fprintf(outFile, "matrix n\n");
    n.print(outFile);

    fprintf(outFile, "matrix o\n");
    o.print(outFile);

    fprintf(outFile, "matrix x\n");
    x.print(outFile);

    fprintf(outFile, "matrix y\n");
    y.print(outFile);

    fprintf(outFile, "matrix z\n");
    z.print(outFile);

    output = x.t();
    fprintf(outFile, "transpose of x\n");
    output.print(outFile);

    output = m.t();
    fprintf(outFile, "transpose of m\n");
    output.print(outFile);

    output = o.t();
    fprintf(outFile, "transpose of o\n");
    output.print(outFile);

    output = n + x;
    fprintf(outFile, "n + x\n");
    output.print(outFile);

    output = m + o;
    fprintf(outFile, "m + o\n");
    output.print(outFile);

    output = n.t() + x;
    fprintf(outFile, "transpose of n + x\n");
    output.print(outFile);

    output = x.t() - n.t();
    fprintf(outFile, "transpose of x - transpose of n\n");
    output.print(outFile);

    output = y * z;
    fprintf(outFile, "y * z\n");
    output.print(outFile);

    output = n ^ x;
    fprintf(outFile, "matrix multiplication n ^ x\n");
    output.print(outFile);

    Mat<> newMat(2,4);
    m.copy(newMat);
    fprintf(outFile, "m.copy(newMat)\n");
    newMat.print(outFile);

    output = y.copy();
    fprintf(outFile, "y.copy()\n");
    output.print(outFile);

    Mat<float> outfloat;
    outfloat = x.copy<float>();
    fprintf(outFile, "casting matrix x to float using copy<float>()\n");
    outfloat.print(outFile);

    x.copy(outfloat);
    fprintf(outFile, "casting matrix x to float using copy(dest)\n");
    outfloat.print(outFile);

    Mat<int> outInt;
    outInt = x.copy<int>() & 1;
    fprintf(outFile, "using bitwise and: x & 1\n");
    outInt.print(outFile);

    outInt = x.copy<int>() | 7;
    fprintf(outFile, "using bitwise or: x | 7\n");
    outInt.print(outFile);

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

    output = z.roi(1,3,0,1,2,4,2,5);
    fprintf(outFile, "z.roi(1,3,0,1,2,4,2,5)\n");
    output.print(outFile);

    output = x.roi(-1);
    fprintf(outFile, "x.roi(-1)\n");
    output.print(outFile);
    
    Mat<bool> mask({true, true, false, true,
                    false, true, false, false}, 2, 4);
    output = m.i(mask);
    fprintf(outFile, "fancy indexing to mask just elements 0,1,3,5 of matrix m\n");
    output.print(outFile);

    output.scalarFill(false);
    m.ito(mask, output);
    fprintf(outFile, "applying the mask using ito\n");
    output.print(outFile);

    Mat<size_t> index({1,3});
    output = y.i(index);
    fprintf(outFile, "fancy indexing to get just rows 1 and 3 of matrix y\n");
    output.print(outFile);

    output = z.i(index);
    fprintf(outFile, "using the same index to get layers 1 and 3 of matrix z\n");
    output.print(outFile);

    output.scalarFill(0);
    z.ito(index, output);
    fprintf(outFile, "applying the indexing using ito\n");
    output.print(outFile);

    fprintf(outFile, "m.T()\n");
    m.T().print(outFile);

    output = Mat<double>::empty_like(m);
    output = output.reshape(m.columns(),m.rows());
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

    output = z + 5;
    fprintf(outFile, "broadcast scalar: z + 5\n");
    output.print(outFile);

    output = 10.0 - x;
    fprintf(outFile, "broadcast scalar: 10 - x\n");
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
    fprintf(outFile, "assignment of those elements to 7s\n");
    a.print(outFile);

    a += b;
    fprintf(outFile, "a += b\n");
    a.print(outFile);

    a -= b;
    fprintf(outFile, "a -= b\n");
    a.print(outFile);

    a *= 2;
    fprintf(outFile, "a *= 2\n");
    a.print(outFile);

    a /= 2;
    fprintf(outFile, "a /= 2\n");
    a.print(outFile);

    output = a.i(index);
    fprintf(outFile, "fancy indexing to get just elements 1 and 3 of matrix a\n");
    output.print(outFile);

    output.scalarFill(0);
    a.ito(index, output);
    fprintf(outFile, "and again using ito\n");
    output.print(outFile);

    output = Mat<double>::broadcast(a,b,Max);
    fprintf(outFile, "broadcast(a,b,Max);\n");
    output.print(outFile);

    output = Mat<double>::broadcast(a,3.0,Max);
    fprintf(outFile, "broadcast(a,3,Max);\n");
    output.print(outFile);

    output = Mat<double>::broadcast(5.0,b,Max);
    fprintf(outFile, "broadcast(5,b,Max);\n");
    output.print(outFile);

    output = Mat<>::zeros(8,8);
    fprintf(outFile, "8x8 matrix of zeros\n");
    output.print(outFile);

    output = Mat<>::ones(3,2,2,5,6);
    fprintf(outFile, "3x3x2x2x2 matrix of ones\n");
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

    fprintf(outFile, "reshape the previous matrix into a 14x2 matrix\n");
    output.reshape(14,2).print(outFile);

    fprintf(outFile, "reshape again into a 2x2x7 matrix\n");
    output.reshape(2,2,7).print(outFile);

    fprintf(outFile, "matrix x as a 1d, 9 element matrix\n");
    x.reshape(9).print(outFile);

    fprintf(outFile, "printing matrix y again as a reminder of its current state\n");
    y.print(outFile);

    output = y.roi(1,3).reshape(6,2);
    fprintf(outFile, "y.roi(1,3).reshape(6,2)\n");
    if(output.memory == y.memory)
        fprintf(outFile, "points to the same data because it is contiguous\n");
    else
        fprintf(outFile,
            "something went wrong, the new matrix has a different memory pointer!\n");
    output.print(outFile);

    output = y.t().reshape(2,12);
    fprintf(outFile, "reshaping the transpose of y to a 2,12 matrix\n");
    if(output.memory != y.memory)
        fprintf(outFile, "points to a copy because it is not contiguous\n");
    else
        fprintf(outFile,
            "something went wrong, the new matrix has the same memory pointer!\n");
    output.print(outFile);

    fprintf(outFile, "reshaping y to a 2,2,3,2 matrix\n");
    output = y.reshape(2,2,3,2);
    output.print(outFile);

    fprintf(outFile, "arange(3)\n");
    Mat<size_t>::arange(3).print(outFile);

    fprintf(outFile, "arange(3,7)\n");
    Mat<size_t>::arange(3,7).print(outFile);

    fprintf(outFile, "arange(3,7,2)\n");
    Mat<size_t>::arange(3,7,2).print(outFile);

    fprintf(outFile, "arange(3,10,3)\n");
    Mat<size_t>::arange(3,10,3).print(outFile);

    fprintf(outFile, "arange(17,5,-4)\n");
    Mat<size_t>::arange(17,5,-4).print(outFile);

    Mat<size_t> temp = Mat<size_t>::arange(1,5,2);
    output = y.i(temp);
    fprintf(outFile, "y.i(Mat<size_t>::arange(1,5,2))\n");
    output.print(outFile);

    temp = Mat<size_t>::arange(1,5,-2);
    fprintf(outFile, "printing arange(1,5,-2), which should just be size 0\n");
    temp.print(outFile);

    double e[15] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    size_t shape[2] = {3,5};
    Mat<double> wrapper = Mat<double>::wrap(e, 2, shape);
    fprintf(outFile, "15 element c-style array wrapped into a 3x5 matrix\n");
    wrapper.print(outFile);
    wrapper.roi(0,3,1,4) *= 2;
    fprintf(outFile, "doubling the center 3 columns\n");
    wrapper.print(outFile);

    Mat<bool> boolMat({true, true, false},1,3);
    Mat<bool> outBool;
    fprintf(outFile, "boolMat:\n");

    boolMat.print(outFile);
    outBool = boolMat && x;
    fprintf(outFile, "boolMat && matrix x\n");
    outBool.print(outFile);

    outBool = x && !boolMat;
    fprintf(outFile, "matrix x && !boolMat\n");
    outBool.print(outFile);

    outBool = boolMat || !x;
    fprintf(outFile, "boolMat || !x\n");
    outBool.print(outFile);

    outBool = x || boolMat;
    fprintf(outFile, "x || boolMat\n");
    outBool.print(outFile);

    outBool = true && y;
    fprintf(outFile, "true && y\n");
    outBool.print(outFile);

    outBool = !y || false;
    fprintf(outFile, "!y || false\n");
    outBool.print(outFile);

    outBool = !boolMat && true;
    fprintf(outFile, "!boolMat && true\n");
    outBool.print(outFile);

    fprintf(outFile, "Checking if any elements of !boolMat are true:\n");
    if((!boolMat).any()) fprintf(outFile, "At least one is true!\n");
    else fprintf(outFile, "Nope, all false\n");
    fprintf(outFile, "Checking if all elements of y are true:\n");
    if(y.all()) fprintf(outFile, "All elements are true!\n");
    else fprintf(outFile, "Nope, at least one is false\n");
    fprintf(outFile, "Checking if any elements of x are greater than 4:\n");
    if((x > 4).any()) fprintf(outFile, "Some are greater!\n");
    else fprintf(outFile, "Nope, none are\n");
    fprintf(outFile, "Checking if all elements of y are less than 20:\n");
    if((y < 20).all()) fprintf(outFile, "All elements are less than 20!\n");
    else fprintf(outFile, "Nope, at least one is greater\n");
    fprintf(outFile, "Checking if any elements of y are equal to 3:\n");
    if((y == 3).any()) fprintf(outFile, "At least one is 3!\n");
    else fprintf(outFile, "Nope, no 3s here\n");
    fprintf(outFile, "Checking if all elements in x are not equal to 8:\n");
    if((x != 8).all()) fprintf(outFile, "Nope, no 8s here\n");
    else fprintf(outFile, "At least one element is an 8!\n");
    fprintf(outFile, "Matrix of bools representing all values in y < 5\n");
    outBool = y < 5;
    outBool.print(outFile);
    fprintf(outFile, "Checking if all elements in of an empty matrix are true:\n");
    if(smallerMat.all()) fprintf(outFile, "Well, nothing is false, so we're good!\n");
    else fprintf(outFile, "Uh oh, we found something false!\n");
    fprintf(outFile, "Checking if any elements in an empty matrix are true:\n");
    if(smallerMat.any()) fprintf(outFile, "Uh oh, we found something true!\n");
    else fprintf(outFile, "Nope, nothing true here.\n");

    fprintf(outFile, "printing the current state of matrix y:\n");
    y.print(outFile);
    
    output = y.i(y < 6);
    fprintf(outFile, "y.i(y < 6)\n");
    output.print(outFile);

    output = y.i(Mat<size_t>::arange(0,3,2));
    fprintf(outFile, "assigning rows 0 and 2 of y to a matrix\n");
    output.print(outFile);

    y.i(y < 1) = 25;
    fprintf(outFile, "y.i(y < 1) = 25\n");
    y.print(outFile);

    y.i(y == 25) = Mat<double>::arange(1,7);
    fprintf(outFile, "replacing the 25s with a new matrix\n");
    y.print(outFile);

    y.i(Mat<size_t>::arange(1,4,2)) = 0;
    fprintf(outFile, "assigning 0 to rows 1 and 3\n");
    y.print(outFile);

    y.i(Mat<size_t>::arange(1,4,2)) = Mat<double>::arange(1,13);
    fprintf(outFile, "replacing rows 1 and 3 with another matrix\n");
    y.print(outFile);

    Mat<double> invertible({3,0,2,2,0,-2,0,1,1},3,3);
    fprintf(outFile, "Invertible Matrix:\n");
    invertible.print(outFile);

    output = invertible^inv(invertible);
    fprintf(outFile, "multiplied by its inverse\n");
    if((output > Mat<>::eye(output.columns()) - 1e-15).all() && (output < Mat<>::eye(output.columns()) + 1e-15).all()){
        Mat<>::eye(output.columns()).print(outFile);
    }
    else
        output.print(outFile);

    Mat<double> hilbert({1.0/1,1.0/2,1.0/3,1.0/4,1.0/5,
                        1.0/2,1.0/3,1.0/4,1.0/5,1.0/6,
                        1.0/3,1.0/4,1.0/5,1.0/6,1.0/7,
                        1.0/4,1.0/5,1.0/6,1.0/7,1.0/8,
                        1.0/5,1.0/6,1.0/7,1.0/8,1.0/9},5,5);
    fprintf(outFile, "The dreaded Hilbert matrix:\n");
    hilbert.print(outFile);

    output = inv(hilbert);
    fprintf(outFile, "inverse of the Hilbert matrix\n");
    output.print(outFile);

    fflush(outFile);
    std::ifstream testFile("matTest/matTestOutput.txt", std::ifstream::ate | std::ifstream::binary);
    std::ifstream testCheck("matTest/expectedMatTestOutput.txt", std::ifstream::ate | std::ifstream::binary);
    if(testFile.tellg() != testCheck.tellg()){
        printf("Test Failure\n");
    }
    else{
        testFile.seekg(0); testCheck.seekg(0);
        if(!std::equal(std::istreambuf_iterator<char>(testFile),
                std::istreambuf_iterator<char>(),
                std::istreambuf_iterator<char>(testCheck)))
            printf("Test Failure\n");
    }

    return 0;
}
