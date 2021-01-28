#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
#include <matMath.h>
#include <type_traits>
using namespace std;

template <class Type>
inline Type Add(Type a, Type b){ return a + b; }

template <class Type>
inline Type Subtract(Type a, Type b){ return a - b; }

template <class Type>
inline Type Multiply(Type a, Type b){ return a * b; }

template <class Type>
inline Type Divide(Type a, Type b){ return a / b; }

template <class Type, class Type2>
inline bool And(Type a, Type2 b){ return static_cast<bool>(a) && static_cast<bool>(b); }

template <class Type, class Type2>
inline bool Or(Type a, Type2 b){ return static_cast<bool>(a) || static_cast<bool>(b); }

template <class Type>
inline bool Equality(Type a, Type b){ return a == b; }

template <class Type>
inline bool Inequality(Type a, Type b){ return a != b; }

template <class Type>
inline bool LessThan(Type a, Type b){ return a < b; }

template <class Type>
inline bool LessThanEqual(Type a, Type b){ return a <= b; }

template <class Type>
inline bool GreaterThan(Type a, Type b){ return a > b; }

template <class Type>
inline bool GreaterThanEqual(Type a, Type b){ return a >= b; }

template <class Type>
class MatIter;
template <class Type>
class Const_MatIter;

template <class Type = double>
class Mat {
    friend class MatIter<Type>;
    friend class Const_MatIter<Type>;

    public:

    typedef MatIter<Type> iterator;
    typedef Const_MatIter<Type> const_iterator;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;
    typedef Type value_type;
    typedef Type * pointer;
    typedef Type & reference;

    long ndims = 2;
    size_type* dims;
    size_type* strides;
    Type* memory; 
    Type* data;
    int32_t* refCount;
    void* customTypeData = NULL;
    void (*customDestructor)(Mat<Type>*, void*) = 0;

    void errorCheck(bool e, const char* message) const{
        if(e){
            fprintf(stderr, "%s\n", message);
            exit(1);
        }
        return;
    }

    iterator begin(){
        return iterator(*this, 0);
    }

    iterator end(){
        return iterator(*this, size());
    }

    const_iterator begin() const{
        return const_iterator(*this, 0);
    }

    const_iterator end() const{
        return const_iterator(*this, size());
    }

    size_type size() const{
        if(ndims == 0) return 0;
        size_type result = dims[0];
        for(long i = 1; i < ndims; i++){
            result *= dims[i];
        }
        return result;
    }

    size_type rows() const{
        errorCheck(ndims < 2, "1d matrix has no rows");
        return this->dims[ndims - 2];
    }

    size_type columns() const{
        return this->dims[ndims - 1];
    }

    template<typename... arg>
    bool inbounds(const arg... ind){
        size_type temp[sizeof...(arg)] = {(static_cast<size_type>(ind))...};
        for(int i = 0; i < ndims; i++){
            if(temp[i] >= dims[i] || temp[i] < 0) return false;
        }
        return true;
    }

    bool isContiguous() const{
        size_type check = 1;
        for(long i = ndims - 1; i >= 0; i--){
            if(strides[i] != check) return false;
            check *= dims[i];
        }
        return true;
    }

    Mat(){
        refCount = new int32_t;
        *refCount = 1;

        ndims = 1;
        dims = new size_type[ndims];
        dims[0] = 1;

        strides = new size_type[ndims];
        strides[0] = 1;

        memory = new Type[1];
        data = memory;
    }

    template<typename... arg>
    Mat(const arg... ind){
        refCount = new int32_t;
        *refCount = 1;

        ndims = sizeof...(arg);
        errorCheck(ndims > 32, "Mat constructed with too many arguments");

        dims = new size_type[ndims];
        size_type temp[sizeof...(arg)] = {(static_cast<size_type>(ind))...};
        for(long i = 0; i < ndims; i++){
            dims[i] = temp[i];
        }

        strides = new size_type[ndims];
        strides[ndims-1] = 1;
        for(long i = 1, j = ndims-2; i < ndims; i++, j--){
            strides[j] = strides[j+1]*dims[i];
        }
     
        memory = new Type[size()];
        data = memory;
    }

    Mat(std::initializer_list<Type> list){
        refCount = new int32_t;
        *refCount = 1;

        ndims = 1;
        dims = new size_type[ndims];
        dims[0] = list.size();

        strides = new size_type[ndims];
        strides[0] = 1;

        memory = new Type[list.size()];
        data = memory;
        size_type i = 0;
        for(auto elem : list){
            data[i] = elem;
            i++;
        }
    }

    template<typename... arg>
    Mat(std::initializer_list<Type> list, const arg... ind){
        refCount = new int32_t;
        *refCount = 1;

        ndims = sizeof...(arg);
        errorCheck(ndims > 32, "Mat constructed with too many arguments");

        dims = new size_type[ndims];
        size_type temp[sizeof...(arg)] = {(static_cast<size_type>(ind))...};
        for(long i = 0; i < ndims; i++){
            dims[i] = temp[i];
        }
        errorCheck(list.size() != size(),
            "Initializer list size inconsistent with dimensions");

        strides = new size_type[ndims];
        strides[ndims-1] = 1;
        for(long i = 1, j = ndims-2; i < ndims; i++, j--){
            strides[j] = strides[j+1]*dims[i];
        }
     
        memory = new Type[size()];
        data = memory;
        size_type i = 0;
        for(auto elem : list){
            data[i] = elem;
            i++;
        }
    }

    Mat(const Mat& b){
        refCount = b.refCount;
        (*refCount)++;
        ndims = b.ndims;
        dims = new size_type[ndims];
        for(long i = 0; i < ndims; i++){
            dims[i] = b.dims[i];
        }
        strides = new size_type[ndims];
        for(long i = 0; i < ndims; i++){
            strides[i] = b.strides[i];
        }
        memory = b.memory;
        data = b.data;
        customDestructor = b.customDestructor;
        customTypeData = b.customTypeData;
    }

    ~Mat(){
        if(customDestructor){
            delete []dims;
            delete []strides;
            customDestructor(this, customTypeData);
            return;
        }
        (*refCount)--;
        errorCheck(*refCount < 0,
            "Reference counter is negative somehow");
        if(*refCount == 0){
            delete refCount;
            if(customTypeData == NULL)
                delete []memory;
        }
        delete []dims;
        delete []strides;
    }

    template<typename... arg>
    Type& operator() (const arg... ind){
        size_type temp[sizeof...(arg)] = {(static_cast<size_type>(ind))...};
        size_type offset = 0;
        for(long i = 0; i < ndims; i++){
            offset += temp[i]*strides[i];
        }
        return data[offset];
    }

    template<typename... arg>
    const Type& operator() (const arg... ind) const{
        size_type temp[sizeof...(arg)] = {(static_cast<size_type>(ind))...};
        size_type offset = 0;
        for(long i = 0; i < ndims; i++){
            offset += temp[i]*strides[i];
        }
        return data[offset];
    }

    Type& operator() (iterator i){
        return data[i.position];
    }

    Mat& operator= (const Mat &b){
        this->~Mat<Type>();
        refCount = b.refCount;
        (*refCount)++;
        ndims = b.ndims;
        dims = new size_type[ndims];
        for(long i = 0; i < ndims; i++){
            dims[i] = b.dims[i];
        }
        strides = new size_type[ndims];
        for(long i = 0; i < ndims; i++){
            strides[i] = b.strides[i];
        }
        memory = b.memory;
        data = b.data;
        customDestructor = b.customDestructor;
        customTypeData = b.customTypeData;
        return *this;
    }

    Mat<Type>& operator= (Type scalar){
        for(auto& i: *this){
            i = scalar;
        }
        return *this;
    }
    
    Mat<Type> operator+(const Mat<Type> &b){
        return broadcast(b, Add<Type>);
    }

    Mat<Type> operator+(Type b){
        return broadcast(b, Add<Type>);
    }

    void operator +=(const Mat<Type> &b){
        broadcast(b, Add<Type>, *this);
    }

    void operator +=(Type b){
        broadcast(b, Add<Type>, *this);
    }

    Mat<Type> operator-(const Mat<Type> &b){
        return broadcast(b, Subtract<Type>);
    }

    Mat<Type> operator-(Type b){
        return broadcast(b, Subtract<Type>);
    }

    void operator -=(const Mat<Type> &b){
        broadcast(b, Subtract<Type>, *this);
    }

    void operator -=(Type b){
        broadcast(b, Subtract<Type>, *this);
    }

    Mat<Type> operator*(const Mat<Type> &b){
        return broadcast(b, Multiply<Type>);
    }

    Mat<Type> operator*(Type b){
        return broadcast(b, Multiply<Type>);
    }

    void operator *=(const Mat<Type> &b){
        broadcast(b, Multiply<Type>, *this);
    }

    void operator *=(Type b){
        broadcast(b, Multiply<Type>, *this);
    }

    Mat<Type> operator/(const Mat<Type> &b){
        return broadcast(b, Divide<Type>);
    }

    Mat<Type> operator/(Type b){
        return broadcast(b, Divide<Type>);
    }

    void operator /=(const Mat<Type> &b){
        broadcast(b, Divide<Type>, *this);
    }

    void operator /=(Type b){
        broadcast(b, Divide<Type>, *this);
    }

    template<class Type2>
    Mat<bool> operator&(const Mat<Type2> &b){
        return broadcast(b, And<Type,Type2>);
    }

    Mat<bool> operator&(bool b){
        Mat<bool> temp({b},1);
        return broadcast(b, And<Type,bool>);
    }

    template<class Type2>
    Mat<bool> operator|(const Mat<Type2> &b){
        return broadcast(b, Or<Type,Type2>);
    }

    Mat<bool> operator|(bool b){
        Mat<bool> temp({b},1);
        return broadcast(b, Or<Type,bool>);
    }

    Mat<bool> operator!(); // defined below

    Mat<bool> operator==(const Mat<Type> b){
        return broadcast(b, Equality<Type>);
    }

    Mat<bool> operator==(Type b){
        Mat<Type> temp({b},1);
        return broadcast(b, Equality<Type>);
    }

    Mat<bool> operator!=(const Mat<Type> b){
        return broadcast(b, Inequality<Type>);
    }

    Mat<bool> operator!=(Type b){
        Mat<Type> temp({b},1);
        return broadcast(b, Inequality<Type>);
    }

    Mat<bool> operator<(const Mat<Type> b){
        return broadcast(b, LessThan<Type>);
    }

    Mat<bool> operator<(Type b){
        Mat<Type> temp({b},1);
        return broadcast(b, LessThan<Type>);
    }

    Mat<bool> operator<=(const Mat<Type> b){
        return broadcast(b, LessThanEqual<Type>);
    }

    Mat<bool> operator<=(Type b){
        Mat<Type> temp({b},1);
        return broadcast(b, LessThanEqual<Type>);
    }

    Mat<bool> operator>(const Mat<Type> b){
        return broadcast(b, GreaterThan<Type>);
    }

    Mat<bool> operator>(Type b){
        Mat<Type> temp({b},1);
        return broadcast(b, GreaterThan<Type>);
    }

    Mat<bool> operator>=(const Mat<Type> b){
        return broadcast(b, GreaterThanEqual<Type>);
    }

    Mat<bool> operator>=(Type b){
        Mat<Type> temp({b},1);
        return broadcast(b, GreaterThanEqual<Type>);
    }

    bool all(){
        for(auto i : *this){
            if(static_cast<bool>(i) == false) return false;
        }
        return true;
    }

    bool any(){
        for(auto i : *this){
            if(static_cast<bool>(i) == true) return true;
        }
        return false;
    }

    template<class Type2, class Type3>
    Mat<Type3> broadcast(const Mat<Type2> &b, Type3 (*f)(Type, Type2)){
        Mat<Type3> out;
        delete[] out.dims;
        delete[] out.strides;

        size_type* big_dim;
        size_type* small_dim;
        long dimdiff;
        if(ndims >= b.ndims){
            big_dim = dims;
            small_dim = b.dims;
            dimdiff = ndims - b.ndims;
            out.ndims = ndims;
            out.dims = new size_type[ndims];
            out.strides = new size_type[ndims];
        }
        else{
            big_dim = b.dims;
            small_dim = dims;
            dimdiff = b.ndims - ndims;
            out.ndims = b.ndims;
            out.dims = new size_type[b.ndims];
            out.strides = new size_type[b.ndims];
        }
        errorCheck(ndims != 1 && b.ndims != 1 && dimdiff != 0,
            "operands could not be broadcast together");

        for(long i = 0; i < out.ndims; i++){
            if(i < dimdiff) out.dims[i] = big_dim[i];
            else{
                if(big_dim[i] > small_dim[i - dimdiff]) out.dims[i] = big_dim[i];
                else out.dims[i] = small_dim[i - dimdiff];
            }
        }

        out.strides[ndims-1] = 1;
        for(long i = 1, j = out.ndims-2; i < out.ndims; i++, j--){
            out.strides[j] = out.strides[j+1]*out.dims[i];
        }

        delete[] out.memory;
        out.memory = new Type3[out.size()];
        out.data = out.memory;

        broadcast(b, f, out);
        return out;
    }

    template<class Type2, class Type3>
    void broadcast(const Mat<Type2> &b, Type3 (*f)(Type, Type2), Mat<Type3> &out){
        if(ndims >= b.ndims){
            for(long n = 0; n < ndims; n++){
                if(n < b.ndims && dims[n] > b.dims[n]){
                    errorCheck(b.dims[n] != 1 || out.dims[n] != dims[n],
                                "frames not aligned");
                }
                else if(n < b.ndims && dims[n] < b.dims[n]){
                    errorCheck(dims[n] != 1 || out.dims[n] != b.dims[n],
                                "frames not aligned");
                }
                else{
                    errorCheck(out.dims[n] != dims[n],
                                "broadcast output matrix frame misaligned");
                }
            }
        }
        else{
            for(long n = 0; n < b.ndims; n++){
                if(n < ndims && dims[n] > b.dims[n]){
                    errorCheck(b.dims[n] != 1 || out.dims[n] != dims[n],
                                "frames not aligned");
                }
                else if(n < ndims && dims[n] < b.dims[n]){
                    errorCheck(dims[n] != 1 || out.dims[n] != b.dims[n],
                                "frames not aligned");
                }
                else{
                    errorCheck(out.dims[n] != b.dims[n],
                                "broadcast output matrix frame misaligned");
                }
            }
        }

        size_type caststrideA[32];
        size_type caststrideB[32];
        for(long i = 0; i < out.ndims; i++){
            if(ndims < out.ndims - i || dims[i - (out.ndims - ndims)] == 1)
                caststrideA[i] = 0;
            else caststrideA[i] = strides[i - (out.ndims - ndims)];
            if(b.ndims < out.ndims - i || b.dims[i - (out.ndims - b.ndims)] == 1)
                caststrideB[i] = 0;
            else caststrideB[i] = b.strides[i - (out.ndims - b.ndims)];
        }

        size_type posA = 0, posB = 0, posOut = 0;
        size_type dimind[32];
        for(long i = 0; i < out.ndims; i++){
            dimind[i] = 0;
        }
        for(size_type i = 0; i < out.size(); i++){
            out.data[posOut] = f(this->data[posA],b.data[posB]);
            for(long j = out.ndims-1; j >= 0; j--){
                dimind[j]++;
                if(dimind[j] != out.dims[j]){
                    posA += caststrideA[j];
                    posB += caststrideB[j];
                    posOut += out.strides[j];
                    break;
                }
                else{
                    dimind[j] = 0;
                    posA -= caststrideA[j]*(out.dims[j]-1);
                    posB -= caststrideB[j]*(out.dims[j]-1);
                    posOut -= out.strides[j]*(out.dims[j]-1);
                }
            }
        }
    }

    template<class Type2, class Type3>
    Mat<Type3> broadcast(Type2 b, Type3 (*f)(Type, Type2)){
        Mat<Type2> temp({b},1);
        return broadcast(temp, *f);
    }

    template<class Type2, class Type3>
    void broadcast(Type2 b, Type3 (*f)(Type, Type2), Mat<Type3> &out){
        Mat<Type2> temp({b},1);
        return broadcast(temp, *f, out);
    }

    Mat operator- (){
        Mat<Type> temp({-1},1);
        return broadcast(temp, Multiply<Type>);
    }

    Mat operator^ (const Mat<Type> &b){
        errorCheck(ndims != 2 || b.ndims != 2,
            "Matrix multiply only available on 2d matrices");
        errorCheck(columns() != b.rows(), "Matrix size mismatch");
        Mat<Type> result(rows(),b.columns());
        Type sum;
        for(size_type x = 0; x < rows(); x++){
            for(size_type i = 0; i<b.columns();i++){
                sum = 0;
                for(size_type n = 0; n < columns(); n++){
                    sum += operator()(x,n)*b(n,i);
                }
                result(x,i) = sum;
            }
        }
        return result;
    }

    template<typename... arg>
    Mat<Type> roi(const arg... ind){
        Mat<Type> out(*this);
        if(sizeof...(arg) == 0) return out;
        errorCheck(sizeof...(arg) > static_cast<size_type>(2*out.ndims),
                    "too many arguments for roi function");
        int temp[sizeof...(arg)] = {(static_cast<int>(ind))...};
        
        for(long i = 0; i < out.ndims; i++){
            if(static_cast<size_type>(2*i) >= sizeof...(arg)) break;
            else if(static_cast<size_type>(2*i)+1 >= sizeof...(arg)){
                if(temp[2*i] == -1) temp[2*i] = 0;
                errorCheck(temp[(2*i)] < 0
                        || static_cast<size_type>(temp[2*i]) > out.dims[i],
                        "roi shape mismatch");
                out.dims[i] -= temp[2*i];
                out.data += temp[2*i]*out.strides[i];
            }
            else{
                if(temp[(2*i)+1] == -1) temp[(2*i)+1] = out.dims[i];
                if(temp[2*i] == -1) temp[2*i] = 0;
                errorCheck(temp[(2*i)+1] < 0 || temp[(2*i)] < 0
                        || static_cast<size_type>(temp[(2*i)+1]) > out.dims[i]
                        || static_cast<size_type>(temp[2*i]) > out.dims[i],
                        "roi shape mismatch");
                out.dims[i] = temp[(2*i)+1] - temp[2*i];
                out.data += temp[2*i]*out.strides[i];
            }
        }
        return out;
    }

    //i has 4 versions depending on whether the given parameter is a boolean
    //mask or a list of indices. The default parameter in the indexed version
    //simply causes substitution to fail when a floating point matrix is passed.
    Mat<Type> i(Mat<bool> &mask);
    template<typename Type2>
    Mat<Type> i(Mat<Type2> &indices,
                typename std::enable_if<std::is_integral<Type2>::value>::type* = 0);
    void ito(Mat<bool> &mask, Mat<Type> &out);
    template<typename Type2>
    void ito(Mat<Type2> &indices, Mat<Type> &out,
                typename std::enable_if<std::is_integral<Type2>::value>::type* = 0);

    Mat T(Mat& dest){
        errorCheck(ndims != 2,
            "transpose may only be used on 2d matrix");
        errorCheck(memory == dest.memory,
            "Source and destination matrix share same backing data");
        t().copy(dest);
        return dest;
    }

    Mat T(){
        errorCheck(ndims != 2, "transpose may only be used on 2d matrix");
        if(rows() == columns()){
            Type temp;
            for(size_type i=0; i<rows(); i++){
                for(size_type j=i+1; j<columns(); j++){
                    temp = operator()(i,j);
                    operator()(i,j) = operator()(j,i);
                    operator()(j,i) = temp;
                }
            }
        }
        else if(isContiguous()){
            Mat<Type> clone(rows(), columns());
            copy(clone);
            reshape(columns(), rows());
            for(size_type i = 0; i < columns(); i++){
                for(size_type j = 0; j < rows(); j++){
                    operator()(j,i) = clone(i,j);
                }
            }
        }
        else{
            errorCheck(true,
                "transpose may only be used on square or continuous matrices");
        }
        return *this;
    }

    Mat t() const{
        errorCheck(ndims != 2, "transpose may only be used on 2d matrix");
        Mat<Type> dest(*this);
        dest.strides[0] = strides[1];
        dest.strides[1] = strides[0];
        dest.dims[0] = dims[1];
        dest.dims[1] = dims[0];
        return dest;
    }

    void print(){
        size_t n = 0;
        for(auto i : *this){
            printf("%g", (double)i);
            n++;
            if(n%columns() != 0) printf(", ");
            else printf("\n");
        }
        return;
    }

    void print(FILE* output){
        size_t n = 0;
        for(auto i : *this){
            fprintf(output, "%g", (double)i);
            n++;
            if(n%columns() != 0) fprintf(output, ", ");
            else fprintf(output, "\n");
        }
        return;
    }

    template<class newType = Type>
    Mat<newType> copy() const{
        Mat<newType> dest;
        if(ndims == 2){
            Mat<newType> temp(rows(),columns());
            dest = temp;
        }
        else{
            Mat<newType> temp(columns());
            dest = temp;
        }
        size_t n = 0;
        for(auto i : *this){
            dest.data[n] = static_cast<newType>(i);
            n++;
        }
        return dest;
    }

    template<class newType>
    void copy(Mat<newType>& dest) const{
        errorCheck(dest.ndims != ndims,
            "Matrix dimension mismatch during copy");
        for(long i = 0; i > dest.ndims; i++){
            errorCheck(dest.dims[i] != dims[i], "Matrix size mismatch");
        }
        if(ndims == 2){
            size_t m = 0;
            size_t n = 0;
            for(auto i : *this){
                dest(m,n) = static_cast<newType>(i);
                n++;
                if(n == columns()){
                    n = 0;
                    m++;
                }
            }
        }
        else if(ndims == 1){
            size_t n = 0;
            for(auto i : *this){
                dest(n) = static_cast<newType>(i);
                n++;
            }
        }
        else errorCheck(true, "n-dimensional copy not yet implemented");
        return;
    }

    void scalarFill(Type x){
        for(auto& i : *this){
            i = x;
        }
    }

    void reshape(int new_dim1 = -1){
        errorCheck(!isContiguous(),
            "Cannot reshape non-contiguous matrix");
        errorCheck(new_dim1 < -1,
            "matrix cannot have negative dimensions");
        if(new_dim1 == -1) new_dim1 = size();
        else errorCheck(size() != static_cast<size_type>(new_dim1),
                        "new shape size mismatch");

        if(ndims == 1) return;
        else{
            ndims = 1;
            delete[] dims;
            delete[] strides;
            dims = new size_type[ndims];
            dims[0] = new_dim1;
            strides = new size_type[ndims];
            strides[0] = 1;
        }
        return;
    }

    void reshape(int new_dim1, int new_dim2){
        errorCheck(!isContiguous(),
            "Cannot reshape non-contiguous matrix");
        errorCheck(new_dim1 < -1 || new_dim2 < -1,
            "matrix cannot have negative dimensions");
        errorCheck(new_dim1 == -1 && new_dim2 == -1,
            "only one argument of reshape can be -1");
        if(new_dim1 == -1) new_dim1 = size()/new_dim2;
        else if(new_dim2 == -1) new_dim2 = size()/new_dim1;
        else errorCheck(size() !=
                static_cast<size_type>(new_dim1) * static_cast<size_type>(new_dim2),
                "new shape size mismatch");

        if(ndims == 2){
            dims[0] = static_cast<size_type>(new_dim1);
            dims[1] = static_cast<size_type>(new_dim2);
            strides[0] = static_cast<size_type>(new_dim2);
        }
        else{
            ndims = 2;
            delete[] dims;
            delete[] strides;
            dims = new size_type[ndims];
            dims[0] = new_dim1;
            dims[1] = new_dim2;
            strides = new size_type[ndims];
            strides[1] = 1;
            strides[0] = new_dim2;
        }
        return;
    }

    static Mat<Type> wrap(Type* data, long new_ndims,
                            size_type* new_dims, size_type* strides = NULL){
        Mat<Type> result;
        delete[] result.dims;
        delete[] result.strides;
        delete[] result.memory;
        result.ndims = new_ndims;
        result.dims = new size_type[result.ndims];
        result.strides = new size_type[result.ndims];
        for(long i = 0; i < result.ndims; i++){
            result.dims[i] = new_dims[i];
            if(strides != NULL) result.strides[i] = strides[i];
        }
        if(strides == NULL){
            result.strides[result.ndims-1] = 1;
            if(result.ndims == 2) result.strides[0] = result.dims[1];
        }
        result.customTypeData = data;
        result.memory = data;
        result.data = data;
        return result;
    }

    static Mat<Type> wrap(Type* data, long new_ndims,
                        size_type* new_dims, size_type* new_strides, 
                        int64_t* ref, void (*destructor)(Mat<Type>*, void*),
                        void* arr){
        Mat<Type> result;
        delete[] result.dims;
        delete[] result.strides;
        delete[] result.memory;
        delete result.refCount;
        result.refCount = reinterpret_cast<int32_t*>(ref);
        (*result.refCount)++;
        result.ndims = new_ndims;
        result.dims = new size_type[result.ndims];
        result.strides = new size_type[result.ndims];
        for(long i = 0; i < result.ndims; i++){
            result.dims[i] = new_dims[i];
            result.strides[i] = new_strides[i];
        }
        result.memory = data;
        result.data = data;
        result.customTypeData = arr;
        result.customDestructor = destructor;
        return result;
    }

    static Mat<Type> wrap(Type* data, long new_ndims,
                        size_type* new_dims, size_type* new_strides, 
                        int32_t* ref, void (*destructor)(Mat<Type>*, void*),
                        void* arr){
        Mat<Type> result;
        delete[] result.dims;
        delete[] result.strides;
        delete[] result.memory;
        delete result.refCount;
        result.refCount = ref;
        (*result.refCount)++;
        result.ndims = new_ndims;
        result.dims = new size_type[result.ndims];
        result.strides = new size_type[result.ndims];
        for(long i = 0; i < result.ndims; i++){
            result.dims[i] = new_dims[i];
            result.strides[i] = new_strides[i];
        }
        result.memory = data;
        result.data = data;
        result.customTypeData = arr;
        result.customDestructor = destructor;
        return result;
    }

    static Mat zeros(size_type a){
        Mat result(a);
        for(auto& i: result){
            i = 0;
        }
        return result;
    }

    static Mat zeros(size_type a, size_type b){
        Mat result(a,b);
        for(auto& i: result){
            i = 0;
        }
        return result;
    }

    static Mat zeros_like(const Mat a){
        Mat result(a.rows(),a.columns());
        for(auto& i: result){
            i = 0;
        }
        return result;
    }

    static Mat ones(size_type a, size_type b){
        Mat result(a,b);
        for(auto& i: result){
            i = 1;
        }
        return result;
    }

    static Mat ones(size_type a){
        Mat result(a);
        for(auto& i: result){
            i = 1;
        }
        return result;
    }

    static Mat ones_like(const Mat a){
        Mat result(a.rows(),a.columns());
        for(auto& i: result){
            i = 1;
        }
        return result;
    }

    static Mat empty_like(const Mat a){
        Mat result(a.rows(),a.columns());
        return result;
    }

    static Mat eye(size_type a){
        Mat result(a,a);
        for(size_type i = 0; i < a; i++){
            for(size_type j = 0; j < a; j++){
                if(i == j) result(i,j) = 1;
                else result(i,j) = 0;
            }
        }
        return result;
    }

    static Mat eye(size_type a, size_type b, int k = 0){
        Mat result(a,b);
        if(k < 0){
            k *= -1;
            for(size_type i = 0; i < a; i++){
                for(size_type j = 0; j < b; j++){
                    if(i-static_cast<size_type>(k) == j) result(i,j) = 1;
                    else result(i,j) = 0;
                }
            }
        }
        else{
            for(size_type i = 0; i < a; i++){
                for(size_type j = 0; j < b; j++){
                    if(i+static_cast<size_type>(k) == j) result(i,j) = 1;
                    else result(i,j) = 0;
                }
            }
        }
        return result;
    }

    template<class Type2, class Type3>
    static Mat<Type3> broadcast(Mat<Type>& a, Type2 b, Type3 (*f)(Type, Type2)){
        return a.broadcast(b, f);
    }

    template<class Type2, class Type3>
    static Mat<Type3> broadcast(Type a, Mat<Type2>& b, Type3 (*f)(Type, Type2)){
        b.errorCheck(b.ndims > 2, "n-dimensional broadcast not yet implemented.");
        if(b.ndims == 2){
            Mat<Type> temp({a},1,1);
            return temp.broadcast(b, f);
        }
        else{
            Mat<Type> temp({a},1);
            return temp.broadcast(b, f);
        }
    }
    
    template<class Type2, class Type3>
    static Mat<Type3> broadcast(Mat<Type>& a, Mat<Type2>& b, Type3 (*f)(Type, Type2)){
        return a.broadcast(b, f);
    }
};

template <class Type>
class MatIter{
    public:

    Mat<Type>& matrix;
    size_t position = 0, index = 0;
    size_t dimind[32];

    MatIter(Mat<Type>& mat, size_t ind) : matrix(mat){
        for(int i = 0; i < matrix.ndims; i++){
            dimind[i] = 0;
        }
        if(ind == matrix.size()){
            index = ind;
            position = matrix.size();
            for(int i = 0; i < matrix.ndims; i++){
                position *= matrix.strides[i];
            }
        }
        while(index < ind){
            (*this)++;
        }
    }
    
    bool operator==(MatIter b){
        matrix.errorCheck(matrix.data != b.matrix.data,
            "Comparison between iterators of different matrices");
        if(index == b.index) return true;
        else return false;
    }

    bool operator!=(MatIter b){
        matrix.errorCheck(matrix.data != b.matrix.data,
            "Comparison between iterators of different matrices");
        if(index != b.index) return true;
        else return false;
    }

    MatIter& operator++(){
        index++;
        for(int i = matrix.ndims-1; i >= 0; i--){
            dimind[i]++;
            if(dimind[i] != matrix.dims[i]){
                position += matrix.strides[i];
                break;
            }
            else{
                dimind[i] = 0;
                position -= matrix.strides[i]*(matrix.dims[i]-1);
            }
        }
        return *this;
    }

    MatIter operator++(int){
        MatIter<Type> clone(*this);
        index++;
        for(int i = matrix.ndims-1; i >= 0; i--){
            dimind[i]++;
            if(dimind[i] != matrix.dims[i]){
                position += matrix.strides[i];
                break;
            }
            else{
                dimind[i] = 0;
                position -= matrix.strides[i]*(matrix.dims[i]-1);
            }
        }
        return clone;
    }

    Type & operator*(){
        return matrix.data[position];
    }
};

template <class Type>
class Const_MatIter{
    public:

    const Mat<Type>& matrix;
    size_t position = 0, index = 0;
    size_t dimind[32];

    Const_MatIter(const Mat<Type>& mat, size_t ind) : matrix(mat){
        for(int i = 0; i < matrix.ndims; i++){
            dimind[i] = 0;
        }
        if(ind == matrix.size()){
            index = ind;
            position = matrix.size();
            for(int i = 0; i < matrix.ndims; i++){
                position *= matrix.strides[i];
            }
        }
        while(index < ind){
            (*this)++;
        }
    }

    bool operator==(Const_MatIter b){
        matrix.errorCheck(matrix.data != b.matrix.data,
            "Comparison between iterators of different matrices");
        if(index == b.index) return true;
        else return false;
    }

    bool operator!=(Const_MatIter b){
        matrix.errorCheck(matrix.data != b.matrix.data,
            "Comparison between iterators of different matrices");
        if(index != b.index) return true;
        else return false;
    }

    Const_MatIter& operator++(){
        index++;
        for(int i = matrix.ndims-1; i >= 0; i--){
            dimind[i]++;
            if(dimind[i] != matrix.dims[i]){
                position += matrix.strides[i];
                break;
            }
            else{
                dimind[i] = 0;
                position -= matrix.strides[i]*(matrix.dims[i]-1);
            }
        }
        return *this;
    }

    Const_MatIter operator++(int){
        Const_MatIter<Type> clone(*this);
        index++;
        for(int i = matrix.ndims-1; i >= 0; i--){
            dimind[i]++;
            if(dimind[i] != matrix.dims[i]){
                position += matrix.strides[i];
                break;
            }
            else{
                dimind[i] = 0;
                position -= matrix.strides[i]*(matrix.dims[i]-1);
            }
        }
        return clone;
    }

    const Type & operator*(){
        return matrix.data[position];
    }
};

template<class Type>
Mat<Type> operator+(Type a, Mat<Type> &b){
    return b.broadcast(a, Add<Type>);
}

template<class Type>
Mat<Type> operator-(Type a, Mat<Type> &b){
    return Mat<Type>::broadcast(a, b, Subtract<Type>);
}

template<class Type>
Mat<Type> operator*(Type a, Mat<Type> &b){
    return b.broadcast(a, Multiply<Type>);
}

template<class Type>
Mat<Type> operator/(Type a, Mat<Type> &b){
    return Mat<Type>::broadcast(a, b, Divide<Type>);
}

template<class Type>
Mat<bool> operator&(bool a, Mat<Type> &b){
    return b.broadcast(a, And<Type,bool>);
}

template<class Type>
Mat<bool> operator|(bool a, Mat<Type> &b){
    return b.broadcast(a, Or<Type,bool>);
}

template<class Type>
Mat<bool> operator==(Type a, Mat<Type> &b){
    return b.broadcast(a, Equality<Type,bool>);
}

template<class Type>
Mat<bool> operator!=(Type a, Mat<Type> &b){
    return b.broadcast(a, Inequality<Type,bool>);
}

template<class Type>
Mat<bool> operator<(Type a, Mat<Type> &b){
    return Mat<Type>::broadcast(a, b, LessThan<Type>);
}

template<class Type>
Mat<bool> operator<=(Type a, Mat<Type> &b){
    return Mat<Type>::broadcast(a, b, LessThanEqual<Type>);
}

template<class Type>
Mat<bool> operator>(Type a, Mat<Type> &b){
    return Mat<Type>::broadcast(a, b, GreaterThan<Type>);
}

template<class Type>
Mat<bool> operator>=(Type a, Mat<Type> &b){
    return Mat<Type>::broadcast(a, b, GreaterThanEqual<Type>);
}

template<class Type>
Mat<bool> Mat<Type>::operator!(){
    Mat<bool> result(this->copy<bool>());
    for(Mat<bool>::iterator i = result.begin(); i != result.end(); ++i){
        *i = !(*i);
    }
    return result;
}
template<class Type>
Mat<Type> Mat<Type>::i(Mat<bool> &mask){
    errorCheck(mask.rows() != rows(),
        "Matrix shape mismatch in call to i()");
    errorCheck(ndims == 2 && mask.columns() != columns(),
        "Matrix shape mismatch in call to i()");
    size_type newSize = 0;
    for(auto i : mask){
        if(i) newSize++;
    }
    Mat<Type> out(newSize);

    Mat<bool>::iterator j = mask.begin();
    iterator k = out.begin();
    for(iterator i = begin(); i != end(); ++i, ++j){
        if(*j){
            *k = *i;
            ++k;
        }
    }
    return out;
}

template<class Type>
template<class Type2>
Mat<Type> Mat<Type>::i(Mat<Type2> &indices,
                typename std::enable_if<std::is_integral<Type2>::value>::type*){
    if(ndims == 2){
        Mat<Type> out(indices.size(), columns());
        for(size_type i = 0; i < indices.size(); i++){
            for(size_type j = 0; j < columns(); j++){
                out(i,j) = operator()(indices(i),j);
            }
        }
        return out;
    }
    else if(ndims == 1){
        Mat<Type> out(indices.size());
        for(size_type i = 0; i < indices.size(); i++){
            out(i) = operator()(indices(i));
        }
        return out;
    }
    else{
        errorCheck(true, "n-dimensional fancy indexing not yet implemented");
        return Mat<Type>::zeros(0);
    }
}

template<class Type>
void Mat<Type>::ito(Mat<bool> &mask, Mat<Type> &out){
    errorCheck(mask.rows() != rows(),
        "Incorrect number of rows in mask in call to i()");
    errorCheck(ndims == 2 && mask.columns() != columns(),
        "Incorrect number of columns in mask in call to i()");
    errorCheck(out.ndims != 1,
                "output of ito function must be 1-dimensional");
    Mat<Type>::size_type newSize = 0;
    for(auto i : mask){
        if(i) newSize++;
    }
    errorCheck(out.size() < newSize, "insufficient space in output matrix");
    if(out.size() > newSize){
        out = out.roi(0,newSize);
    }
    Mat<bool>::iterator j = mask.begin();
    iterator k = out.begin();
    for(iterator i = begin(); i != end(); ++i, ++j){
        if(*j){
            *k = *i;
            ++k;
        }
    }
    return;
}

template<class Type>
template<class Type2>
void Mat<Type>::ito(Mat<Type2> &indices, Mat<Type> &out,
                typename std::enable_if<std::is_integral<Type2>::value>::type*){
    errorCheck(indices.ndims != 1,
        "Index list should be 1 dimension");
    errorCheck(out.ndims != ndims,
        "inconsistent number of dimensions in output matrix in call to ito");
    errorCheck(ndims == 1 && out.columns() != indices.size(),
        "output matrix shape mismatch - incorrect number of columns");
    if(ndims > 1){
        errorCheck(out.columns() != columns(),
            "output matrix shape mismatch - incorrect number of columns");
        errorCheck(out.rows() != indices.size(),
            "output matrix shape mismatch - incorrect number of rows");
    }
    if(ndims == 2){
        for(size_type i = 0; i < indices.size(); i++){
            for(size_type j = 0; j < columns(); j++){
                out(i,j) = operator()(indices(i),j);
            }
        }
        return;
    }
    else if(ndims == 1){
        for(size_type i = 0; i < indices.size(); i++){
            out(i) = operator()(indices(i));
        }
        return;
    }
    else{
        errorCheck(true, "n-dimensional fancy indexing not yet implemented");
        return;
    }
}
