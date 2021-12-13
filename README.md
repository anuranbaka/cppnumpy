Release ver. ALPHA 3.1 6/10/2021


# CPPNUMPY Matrix Library
This is standalone C++ matrix header library that was built to be byte compatible with Numpy. Included are bindings that correctly propagate memory management and error systems between C++ and Python, so that developers can write code without thinking about where the matrix originated.
Our design goal was to have a library that maps 1:1 to a basic subset of the Numpy API. In the rare case that this isn't possible due to language constraints (such as Python's matrix multiply operator "@" which isn't available in C++) we aim for a close equivalent to minimize mental load when porting code between the two.
Any code written in C++ that takes in our matrices can be compiled with a special header we provide, and the resulting shared library will be importable as a module in CPython. The functions and classes will take and return Numpy arrays wherever our matrix class was used.

Generally speaking, when writing on the C++ side, functions behave just like in NumPy, except for the following changes:
- The operator "^" is overloaded to perform matrix multiplication rather than bitwise XOR
- "T" is used for a hard transpose, while "t" is used for a soft transpose
- Due to limitations of C++ syntax, array slicing is replaced by a "region-of-interest" function (roi()).
- Array broadcasting is implemented as a function taking a function pointer
- Fancy indexing is implemented using the i() and ito() functions which construct a special "iMat" object which can be resolved using the Mat constructor.

Additionally, the library contains an interface for custom allocation procedures. Although by default matrix data is heap allocated, using custom procedures could allow for all memory allocation to occur at program start if needed.

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
//indexing is done using '()' so b(0,3) is the element of b at coordinates (0,3) -> 4
//basic convenience functions such as scalarFill can fill a matrix with a given value

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
The following example uses Pybind11 to create a module that can be called from Python:
```
#include <sample.h> //your code is included here
#include <matPybind.h> //allows automatic conversion of Mat <-> np.array (dependent on pybind11)

PYBIND11_MODULE(sample, m){
  m.def("func", &func, "perfoms func on the given array")
  }
```
This allows your C++ functions to be called using a NumPy array as shown in the following Python code:
```
import numpy as np
import sample
map = np.array([[1,2,3]
               ,[4,5,6]])
s.func(map)
print(map)
```
Custom memory allocation is done by defining allocation and deallocation functions:
```
  //specifies how to allocate meta-data such as the list of dims
  void customAllocator_Meta(Mat<T> &mat, void* userdata, long ndim){
    mat.dims = /*address for storing array of size ndim*/;
    mat.strides = /*another address for storing array of size ndim*/;
    
    //mat.base is only allocated on construction. If not NULL you're making a copy of another matrix!
    if(mat.base == NULL)
    {
        mat.base = /*address with enough space for a MatBase<T>*/;
    }
  }
  void customDellocator_Meta(Mat<T> &mat){
    //deallocation procedures for dims and strides go here
    if(mat.base->refCount <= 0){
      //deallocation procedure for mat.base goes here
    }
  }
  //allocation procured for the data contained by the matrix
  void customAllocator_Data(MatBase<T> &base, void* userdata, size_t data_size){
    base.data = /*address with enough space for all data elements*/;
  }
  void customDeallocator_Data(MatBase<T>&){
    //deallocate data here
  }
```
Those functions can then be assigned to an AllocInfo struct which is passed in during construction:
```
  AllocInfo<T> alloc;
  //userdata is a void* that is passed along to the allocation functions for addressing purposes
  //this could be a memory buffer or struct with address information for example
  //if any of these functions are left undefined, then default allocation on the heap will be used
  alloc.userdata = buffer;
  alloc.allocateMeta = customallocatorMeta<T>;
  alloc.deallocateMeta = customdeallocatorMeta<T>;
  alloc.allocateData = customallocatorData<T>;
  alloc.deallocateData = customdeallocatorData<T>;
  
  Mat<T> mat(&alloc, 3, 4); //constructs a 3x3 matrix using the allocation functions defined above
```
  
  
# Installing
The project is built using make. Running "make install" will copy the relevant header files into the default include path. <Mat.h> contains the matrix header and its basic functions, while <matPybind.h> contains the type-caster that allows automatic conversion to/from a NumPy array.

# Running Tests/Examples
Matrix arithmetic is tested in the matTest.cpp, and demonstrates basic matrix math functions as well as the not-so-basic matrix inverse.

Additionally, "floodFill.cpp" presents an example usage of the Mat class for a Flood Fill function. For simplicity's sake, the matrix simply handles a matrix of single-digit numbers as plain text, but demonstrates a potential practical use case for the class.

Both programs are compiled when running "make" in the base directory. If "useLapack=true" is specified in the make statement, the code will link to Lapack and use that instead.

# Functions
###### Template parameter "Type" used to signify the element type
- **(constructor)**: takes an optional initializer list followed by an optional alloc_info struct, and finally the list of dimensions
  - ` Mat(std::initializer_list<Type>, size_t...) `
  - ` Mat(size_t...) `
  - ` Mat(std::initializer_list<Type>, AllocInfo<Type>, size_t...) `
  - ` Mat(AllocInfo<Type>, size_t...) `
  - ` Mat(const Mat&) `
  - ` Mat(const iMat&) `
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
  - ` Type operator(size_t...) `
- **roi** specifies a region of interest and returns a submatrix of a given shape. Parameters are passed in pairs corresponding to each dimension and -1 signifies "to the beginning/end of the dimension". 
  - ` Mat& roi(int...) // e.g. m.roi(start, end, start2, end2, ...)`
- **i** masks an array with a boolean mask or extracts elements with a list of indices. The resulting iMat can be resolved into a new matrix by calling the Mat constructor on it.
  - ` iMat<Type> i(Mat<bool> mask)`
  - ` iMat<Type> i(Mat<IntegralType> indices)`
- **ito** performs the same function as "i" but with an output parameter.
  - ` void ito(Mat<bool> mask, Mat<Type> out) //If the output matrix is too large, the returned matrix will be an roi within out.`
  - ` void ito(Mat<IntegralType> indices, Mat<Type> out)`
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
- **broadcast**: binary operation which applies a given elementwise function and returns resulting matrix. Can be done in-place by adding an output Mat.
  - ` Mat<Type3> broadcast(const Mat<Type2> &operand, Type3 (*function)(Type,Type2)) `
  - ` void broadcast(const Mat<Type2> &operand, Type3 (*function)(Type,Type2), Mat<Type3> output) `
  - ` Mat<Type3> broadcast(Type2 operand, Type3 (*function)(Type,Type2)) `
  - ` void broadcast(Type2 operand, Type3 (*function)(Type,Type2), Mat<Type3> output) `
- **static broadcast**: static version in case matrix must be right operand
  - ` Mat<Type3> broadcast(Type left_operand, const Mat<Type2>& right_operand, Type3 (*function)(Type,Type2)) `
  - ` Mat<Type3> broadcast(Type2 operand, Type3 (*function)(Type,Type2)) `
- **operator^**: matrix multiplication (Not XOR!)
  - ` Mat<Type> operator^(const Mat<Type>&) `
- **T**: performs hard transpose on a 2D matrix, storing it in a destination matrix if given
  - ` Mat<Type> T() `
  - ` Mat<Type> T(Mat&) `
- **t**: performs soft transpose, leaving the underlying data, and changing only how the matrix accesses elements
  - ` void t() `
- **copy**: returns a copy of the matrix that does NOT use the same data pointer, or stores into given destination matrix. Optional template argument allows for type casting.
  - ` Mat<newType> copy<newType>() `
  - ` void copy(Mat<newType>&) `
- **scalarFill**: fills the matrix with a given value
  - ` void scalarFill(Type)  `
- **reshape**: Returns a version of the matrix with new shape. A view is returned if data is contiguous, otherwise a copy is returned. One -1 parameter can be used to infer a dimension.
  - ` Mat<Type> reshape(int...) `
- **inverse**: There is no member function for inverse. See linked in `inv()` under "Additional Non-member Functions" instead
### Boolean Operators
- **operator&&**: elementwise logical AND
  - ` Mat<bool> operator&(const Mat<Type2>&) `
  - ` Mat<bool> operator&(const bool) `
  - ` Mat<bool> operator&(const bool, const Mat<Type2>&) `
- **operator||**: elementwise logical OR
  - ` Mat<bool> operator|(const Mat<Type2>&) `
  - ` Mat<bool> operator|(const bool) `
  - ` Mat<bool> operator|(const bool, const Mat<Type2>&) `
- **operator!**: elementwise logical negation
  - ` Mat<bool> operator!() `
- **operator&**: bitwise AND
  - ` Mat<Type> operator&(const Mat<Type2>&) `
  - ` Mat<Type> operator&(const bool) `
  - ` Mat<Type> operator&(const bool, const Mat<Type2>&) `
- **operator|**: bitwise OR
  - ` Mat<Type> operator|(const Mat<Type2>&) `
  - ` Mat<Type> operator|(const bool) `
  - ` Mat<Type> operator|(const bool, const Mat<Type2>&) `
- **operator~**: bitwise logical negation
  - ` Mat<Type> operator~() `
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
- **wrap**: returns a matrix that uses a given data pointer and array of dimensions. A reference counter, custom destructor and pointer to the external container can be used to wrap arbitrary reference counted containers.
  - ` Mat<Type> wrap(Type* data, long number_of_dimensions, size_type* dimensions, size_type* strides = NULL) `
  - ` Mat<Type> wrap(Type* data, long number_of_dimensions, size_type* dimensions, size_type* strides, int64_t* ref_counter, void (*destructor), void* arr) `
- **zeros**: returns an array of zeros in the given shape
  - ` Mat<Type> zeros() `
- **zeros_like**: returns an array of zeros with the same shape as a given matrix
  - ` Mat<Type> zeros_like(const Mat) `
- **ones**: returns an array of ones in the given shape
  - ` Mat<Type> ones() `
- **ones_like**: returns an array of ones with the same shape as a given matrix
  - ` Mat<Type> ones_like(const Mat) `
- **empty_like**: returns an empty array with the same shape as a given matrix. The new matrix can have a different type using a template.
  - ` Mat<newType> empty_like(const Mat<Type>) `
- **eye**: returns the identity matrix for an NxN matrix, or for a non-square matrix along a given diagonal (default diagonal starts at first element)
  - ` Mat<Type> eye(size_t) `
  - ` Mat<Type> eye(size_t, size_t, int k = 0) `
- **arange** return evenly spaced values within a given interval
  - ` Mat<Type> arange(int stop) `
  - ` Mat<Type> arange(int start, int stop, int step = 1) `
### Additional Non-member Functions
- **inv()**: returns inverse of the given matrix which uses a stable, basic implementation or imports from Lapack
  -  ` Mat<Type> inv(Mat<Type>) `
