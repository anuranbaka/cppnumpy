# CPPNUMPY Matrix Library
This matrix library header is meant to be compatible with numpy, using a similar
structure to the numpy array. Functionality includes operations for matrix arithmetic,
soft transposition, iteration of elements, dimension broadcasting and the designation
of submatrices in a manner similar to numpy's array slicing. The matrix is initialized,
stored and printed in row-major order.

# Usage
The library supports several basic matrix arithmetic operations:
```
#include <Mat.h> //single header library

Mat<double> a(3,3) //creates an uninitialized 3x3 matrix of int
Mat<double> b({1,2,3,4,5,6,7,8},2,4) // creates a 2x4 matrix initialized to the values in curly braces.
Mat<> c({1,2,3},1,3) // empty <> sets contained type to double by default

b(0,3); //returns element at the listed coordinates, in this case: 4

a.scalarFill(1); //fills matrix with the scalar listed

a + c; //broadcasts the matrix c to size 3x3

b.t() //transposes matrix b

b.roi(1,2,0,1) //points to sub-matrix region of interest (start column, end column, start row, end row)





