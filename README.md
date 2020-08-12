# CPPNUMPY Matrix Library
This matrix library header is meant to be compatible with numpy, using a similar
structure to the numpy array. Functionality includes operations for matrix arithmetic,
soft transposition, iteration of elements, dimension broadcasting and the designation of
submatrices in a manner similar to numpy's array slicing. The matrix is initialized, stored
and printed in row-major order.
Generally speaking, it functions just like the NumPy array, except for the following three
changes: the operator "^" is used for matrix multiplication rather than bitwise XOR, "T" is
used for a hard transpose, while "t" is used for a soft transpose, and due to limitations of
C++ syntax, array slicing is replaced by a "region-of-interest" function (roi()).

# Usage
The library supports several basic matrix arithmetic operations:
```
#include <Mat.h>

Mat<double> a(3,3)
//creates an uninitialized 3x3 matrix of int

Mat<double> b({1,2,3,4,5,6,7,8},2,4)
//creates a 2x4 matrix initialized to the values in curly braces.

Mat<> c({1,2,3},1,3)
//empty <> sets contained type to double by default

a.scalarFill(b(0,3));
//fills matrix with the scalar listed, in this case it's an element of b at coordinates (0,3) -> 4

c = b.t();
//assigns a transpose of matrix b to matrix c. Note that the dimensions don't need to match!

Mat<double> subMat = b.roi(1,2,1)
//creates a sub-matrix region of interest (start row, end row, start column, end column)
//this effectively just chops off the first row and column, making a new 1x3 submatrix {6,7,8}

a = a + subMat;
//adds the values in our new submatrix to matrix a
//this causes the submatrix to broadcast out to the correct dimensions (3x3)

a.print();
//prints our new matrix to the console
//in this case we'll see the following 3x3 matrix: {10,11,12,10,11,12,10,11,12}
```

# Installing
The Mat class and its functions are entirely contained within "projects/matlib/Mat.h"
header library. The other files in the repository are all test/example cases.

# Running Tests/Examples
Matrix arithmetic is tested in the Mat_test.cpp, and demonstrates basic
matrix math functions.

Additionally, "projects/Flood Fill/Flood_test" presents an example usage of the Mat class for
a Flood Fill function. For simplicity's sake the matrix simply handles a matrix of single-digit
numbers as plain text, but demonstrates a potential practical use case for the class.

Both programs are compiled when running "make" in the base directory.

# Functions
###### Template parameter "Type" used to signify the element type
- **(constructor)**: takes an optional initializer list followed by dimensions
  - ` Mat(std::initializer_list<Type>, size_t) `
  - ` Mat(std::initializer_list<Type>, size_t, size_t) `
  - ` Mat(size_t) `
  - ` Mat(size_t, size_t) `
  - ` Mat(const Mat&) `
- **(destructor)**: underlying memory is only deleted if it is the last matrix using it
  - ` ~Mat() `
- **operator=**: assignment can take another matrix or a scalar which assigns element-wise
  - ` Mat& operator=(const Mat&) `
  - ` Mat& operator=(const Type) `
- **print**: prints matrix to terminal, or to optional file pointer passed as argument
  - ` void print() `
  - ` void print(FILE*) `
### Iterator/Capacity
- **begin**: returns iterator to beginning
  - ` MatIter begin() `
- **end**: returns iterator to end
  - ` MatIter end() `
- **size**: returns size
  - ` size_t size() `
- **rows**: returns number of rows if matrix has them
  - ` size_t rows() `
- **columns**: returns number of columns
  - ` size_t columns() `
- **inbounds**: returns true if coordinates are within the bounds of the matrix
  - ` bool inbounds() `
- **isContiguous**: returns true if data is aligned in memory
  - ` bool isContiguous() `
### Element Access
- **operator()**: returns element at the given coordinates
  - ` Type operator(size_t) `
  - ` Type operator()(size_t, size_t) `
- **roi** specifies a region of interest and returns a submatrix of a given shape. -1 signifies "to the beginning/end of the dimension"
  - ` Mat& roi(int dim1Start = -1, int dim1End = -1, int dim2Start = -1, int dim2End = -1) `
### Modifiers
- **operator+**: elementwise addition
  - ` Mat<Type> operator+(const Mat<Type>&) `
  - ` Mat<Type> operator+(const Type) `
  - ` Mat<Type> operator+(const Type, const Mat<Type>&) `
- **operator+=**: elementwise addition and assignment
  - ` void operator+=(const Mat<Type>&) `
- **operator-**: elementwise subtraction, or negation when used as unary operator
  - ` Mat<Type> operator-(const Mat<Type>&) `
  - ` Mat<Type> operator-(const Type) `
  - ` Mat<Type> operator-(const Type, const Mat<Type>&) `
  - ` Mat<Type> operator-() `
- **operator-=**: elementwise subtraction and assignment
  - ` void operator-(const Mat<Type>&) `
- **operator\***: elementwise multiplication
  - ` Mat<Type> operator*(const Mat<Type>&) `
  - ` Mat<Type> operator*(const Type) `
  - ` Mat<Type> operator*(const Type, const Mat<Type>&) `
- **operator\*=**: elementwise multiplication and assignment
  - ` void operator*=(const Mat<Type>&) `
- **operator/**: elementwise division
  - ` Mat<Type> operator/(const Mat<Type>&) `
  - ` Mat<Type> operator/(const Type) `
  - ` Mat<Type> operator/(const Type, const Mat<Type>&) `
- **operator/=**: elementwise division and assignment
  - ` void operator/=(const Mat<Type>&) `
- **broadcast**: binary operation which applies a given elementwise function and returns resulting matrix. Imitates NumPy array broadcasting.
  - ` Mat<Type3> broadcast(const Mat<Type2>&, Type3 (*f)(Type,Type2)) `
  - ` Mat<Type3> broadcast(Type2, Type3 (*f)(Type,Type2)) `
- **static broadcast**: static version in case matrix must be right operand
  - ` Mat<Type3> broadcast(Type, const Mat<Type2>&, Type3 (*f)(Type,Type2)) `
  - ` Mat<Type3> broadcast(Type2, Type3 (*f)(Type,Type2)) `
- **operator^**: matrix multiplication (Not XOR!)
  - ` Mat<Type> operator^(const Mat<Type>&) `
- **T**: performs hard transpose, storing them in destination matrix if given
  - ` Mat<Type> T() `
  - ` Mat<Type> T(Mat&) `
- **t**: performs soft transpose, leaving the underlying data, and changing only how the matrix accesses elements
  - ` void t() `
- **copy**: returns a copy of the matrix that does NOT use the same data pointer, or stores into given destination matrix. Optional template argument allows for type casting.
  - ` Mat<newType> copy<newType>() `
  - ` void copy(Mat<newType>&) `
- **scalarFill**: fills a matrix with a given value
  - ` void scalarFill(Type)  `
- **reshape**: sets the matrix dimensions equal to given arguments while preserving element order. One -1 can be used to infer new dimension.
  - ` void reshape(int = -1) `
  - ` void reshape(int, int) `
- **inverse**: non-member function that takes a mat and returns its inverse
  - ` Mat<Type> inverse(Mat<Type>) `
### Boolean Operators
- **operator&**: elementwise AND
  - ` Mat<bool> operator&(const Mat<Type2>&) `
  - ` Mat<bool> operator&(const bool) `
  - ` Mat<bool> operator&(const bool, const Mat<Type2>&) `
- **operator|**: elementwise OR
  - ` Mat<bool> operator|(const Mat<Type2>&) `
  - ` Mat<bool> operator|(const bool) `
  - ` Mat<bool> operator|(const bool, const Mat<Type2>&) `
- **operator!**: elementwise negation
  - ` Mat<bool> operator!() `
- **all**: returns true if no element of matrix is false
  - ` bool all() `
- **any**: returns true if any element of matrix is true
  - ` bool any() `
### Relational Operators
- **operator==, operator!=, operator<, operator<=, operator>, operator>=**: elementwise relational operators
  - ` Mat<bool> operator==(const Mat<Type>&) `
  - ` Mat<bool> operator==(const Type) `
  - ` Mat<bool> operator==(const Type, const Mat<Type>&) `
### Static Functions
- **wrap**: returns a matrix that uses a given data pointer and array of dimensions. An internal reference counter is created if none is given.
  - ` Mat<Type> wrap(size_t size, Type* data, size_type number_of_dimensions, size_type* dimensions) `
  - ` Mat<Type> wrap(size_t size, Type* data, size_type number_of_dimensions, size_type* dimensions, int64_t* ref_counter) `
- **zeros**: returns an array of zeros in the given shape
  - ` Mat<Type> zeros() `
- **zeros_like**: returns an array of zeros with the same shape as a given matrix
  - ` Mat<Type> zeros_like(const Mat) `
- **ones**: returns an array of ones in the given shape
  - ` Mat<Type> ones() `
- **ones_like**: returns an array of ones with the same shape as a given matrix
  - ` Mat<Type> ones_like(const Mat) `
- **empty_like**: returns an empty array with the same shape as a given matrix
  - ` Mat<Type> empty_like(const Mat) `
- **eye**: returns the identity matrix for an NxN matrix, or for a non-square matrix along a given diagonal (default diagonal starts at first element)
  - ` Mat<Type> eye(size_t) `
  - ` Mat<Type> eye(size_t, size_t, int k = 0) `
