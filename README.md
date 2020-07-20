# CPPNUMPY Matrix Library
This matrix library header is meant to be compatible with numpy, using a similar
structure to the numpy array. Functionality includes operations for matrix arithmetic,
soft transposition, iteration of elements, dimension broadcasting and the designation
of submatrices in a manner similar to numpy's array slicing. The matrix is initialized,
stored and printed in row-major order.

# Installing
The Mat class and its functions are entirely contained within "projects/matlib/Mat.h"
header library.

# Running Tests/Examples
Matrix arithmetic is tested in the "projects/Mat_Test/Mat_test.cpp", and demonstrates basic
matrix math functions.

Additionally, "projects/Flood Fill/Flood_test" presents an example usage of the Mat class for
a Flood Fill function. For simplicity's sake the matrix simply handles a matrix of single-digit
numbers as plain text, but demonstrates a potential practical use case for the class.

# Functions
###### Template parameter "Type" used to signify the element type
- **(constructor)**: takes an optional initializer list followed by dimensions
  - Mat(std::initializer_list<Type>, size_t)
  - Mat(size_t)
  - Mat(size_t, size_t)
  - Mat(const Mat&)
- **(destructor)**: underlying memory is only deleted if it is the last matrix using it
  - ~Mat()
- **operator=**: assignment can take another matrix or a scalar which assigns element-wise
  - Mat& operator=(const Mat&)
  - Mat& operator=(const Type)
- **print**: prints matrix to terminal, or to optional file pointer passed as argument
  - void print()
  - void print(FILE*)
### Iterator/Capacity
- **begin**: returns iterator to beginning
  - MatIter begin()
- **end**: returns iterator to end
  - MatIter end()
- **size**: returns size
  - size_t size()
- **rows**: returns number of rows if matrix has them
  - size_t rows()
- **columns**: returns number of columns
  - size_t columns()
- **inbounds**: returns true if coordinates are within the bounds of the matrix
  - bool inbounds()
- **isContiguous**: returns true if data is aligned in memory
  - bool isContiguous()
### Element Access
- **operator()**: returns element at the given coordinates
  - Type operator(size_t)
  - Type operator()(size_t, size_t)
- **roi** specifies a region of interest and returns a submatrix of a given shape. -1 signifies "to the beginning/end of the dimension"
  - Mat& roi(int dim1Start = -1, int dim1End = -1, int dim2Start = -1, int dim2End = -1)
### Modifiers
- **operator+**: elementwise addition
  - Mat operator+(const Mat<Type>&)
- **operator-**: elementwise subtraction, or negation when used as unary operator
  - Mat operator-(const Mat<Type>&)
  - Mat operator-()
- **broadcast**: binary operation which applies a given elementwise function and returns resulting matrix. Imitates NumPy array broadcasting.
  - Mat broadcast(const Mat<Type>&, Type (*f)(Type,Type))
- **operator^**: matrix multiplication
  - Mat operator^(const Mat<Type>&)
- **T**: returns transposed matrix if no arguments are given, otherwise stores transpose into given destination matrix
  - Mat T()
  - void T(Mat&)
- **t**: performs soft transpose, leaving the underlying data, and changing only how the matrix accesses elements
  - void t()
- **copy**: returns a copy of the matrix that does NOT use the same data pointer, or stores into given destination matrix
  - Mat copy()
  - void copy(Mat<Type>&)
- **scalarFill**: fills a matrix with a given value
  - void scalarFill(Type)
- **reshape**: sets the matrix dimensions equal to given arguments while preserving element order. One -1 can be used to infer new dimension.
  - void reshape(int = -1)
  - void reshape(int, int)
### Static Functions
- **wrap**: returns a matrix that uses a given data pointer and array of dimensions. An internal reference counter is created if none is given.
  - Mat<Type> wrap(size_t size, Type* data, size_type number_of_dimensions, size_type* dimensions)
  - Mat<Type> wrap(size_t size, Type* data, size_type number_of_dimensions, size_type* dimensions, int64_t* ref_counter)
- **zeros**: returns an array of zeros in the given shape
  - Mat zeros()
- **zeros_like**: returns an array of zeros with the same shape as a given matrix
  - Mat zeros_like(const Mat)
- **ones**: returns an array of ones in the given shape
  - Mat ones()
- **ones_like**: returns an array of ones with the same shape as a given matrix
  - Mat ones_like(const Mat)
- **empty_like**: returns an empty array with the same shape as a given matrix
  - Mat empty_like(const Mat)
- **eye**: returns the identity matrix for an NxN matrix, or for a non-square matrix along a given diagonal (default diagonal starts at first element)
  - Mat eye(size_t)
  - Mat eye(size_t, size_t, int k = 0)

# Usage
The library supports several basic matrix arithmetic operations:
```
#include <Mat.h>

Mat<double> a(3,3) //creates an uninitialized 3x3 matrix of int
Mat<double> b({1,2,3,4,5,6,7,8},2,4) // creates a 2x4 matrix initialized to the values in curly braces.
Mat<> c({1,2,3},1,3) // empty <> sets contained type to double by default

b(0,3); //returns element at the listed coordinates, in this case: 4

a.scalarFill(1); //fills matrix with the scalar listed

a + c; //broadcasts the matrix c to size 3x3

b.t(); //returns a transpose of matrix b

b.roi(1,2,0,1) //points to sub-matrix region of interest (start column, end column, start row, end row)
```
