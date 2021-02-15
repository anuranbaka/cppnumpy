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

    template<class left, class right, class Type3>
    static void broadcastHelper(const Mat<left>& bigMat, const Mat<right>& smallMat,
                                Mat<Type3>& out,
                                size_t* bigStrides, size_t* smallStrides){
        long dimdiff = bigMat.ndim - smallMat.ndim;
        bigMat.errorCheck(out.ndim != bigMat.ndim,
            "output matrix ndim not equal to broadcasted ndim");
        for(int i = 0; i < dimdiff; i++){
            bigMat.errorCheck(out.dims[i] != bigMat.dims[i],
                "broadcast output matrix shape mismatch");
        }
        for(int i = dimdiff; i < bigMat.ndim; i++){
            if(bigMat.dims[i] != 1 && smallMat.dims[i-dimdiff] != 1){
                bigMat.errorCheck(bigMat.dims[i] != smallMat.dims[i-dimdiff],
                    "operand frames not aligned");
                bigMat.errorCheck(out.dims[i] != bigMat.dims[i],
                    "broadcast output matrix shape mismatch");
            }
            if(bigMat.dims[i] == 1)
                bigMat.errorCheck(out.dims[i] != smallMat.dims[i-dimdiff],
                "broadcast output matrix shape mismatch");
            if(smallMat.dims[i-dimdiff] == 1)
                bigMat.errorCheck(out.dims[i] != bigMat.dims[i],
                "broadcast output matrix shape mismatch");
        }

        for(long i = 0; i < dimdiff; i++){
            bigStrides[i] = bigMat.strides[i];
            smallStrides[i] = 0;
        }
        for(long i = dimdiff; i < bigMat.ndim; i++){
            if(bigMat.dims[i] == 1) bigStrides[i] = 0;
            else bigStrides[i] = bigMat.strides[i];
            if(smallMat.dims[i-dimdiff] == 1) smallStrides[i] = 0;
            else smallStrides[i] = smallMat.strides[i-dimdiff];
        }
    }

    public:
    void buildStrides(){
        errorCheck(ndim == 0, "ndim cannot equal 0");
        if(strides != NULL) delete[] strides;
        strides = new size_type[ndim];
        strides[ndim-1] = 1;
        for(long j = ndim-2; j >= 0; j--){
            strides[j] = strides[j+1]*dims[j+1];
        }
    }

    typedef MatIter<Type> iterator;
    typedef Const_MatIter<Type> const_iterator;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;
    typedef Type value_type;
    typedef Type * pointer;
    typedef Type & reference;

    long ndim = 2;
    size_type* dims;
    size_type* strides = NULL;
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
        if(ndim == 0) return 0;
        size_type result = dims[0];
        for(long i = 1; i < ndim; i++){
            result *= dims[i];
        }
        return result;
    }

    size_type rows() const{
        errorCheck(ndim < 2, "1d matrix has no rows");
        return this->dims[ndim - 2];
    }

    size_type columns() const{
        return this->dims[ndim - 1];
    }

    template<typename... arg>
    bool inbounds(const arg... ind){
        size_type temp[sizeof...(arg)] = {(static_cast<size_type>(ind))...};
        for(int i = 0; i < ndim; i++){
            if(temp[i] >= dims[i] || temp[i] < 0) return false;
        }
        return true;
    }

    bool isContiguous() const{
        size_type check = 1;
        for(long i = ndim - 1; i >= 0; i--){
            if(strides[i] != check) return false;
            check *= dims[i];
        }
        return true;
    }

    Mat(){
        refCount = new int32_t;
        *refCount = 1;

        ndim = 1;
        dims = new size_type[ndim];
        dims[0] = 0;
        buildStrides();

        memory = new Type[0];
        data = memory;
    }

    template<typename... arg>
    Mat(const arg... ind){
        refCount = new int32_t;
        *refCount = 1;

        ndim = sizeof...(arg);
        errorCheck(ndim > 32, "too many dimensions (<=32 allowed)");

        dims = new size_type[ndim]{(static_cast<size_type>(ind))...};
        buildStrides();
     
        memory = new Type[size()];
        data = memory;
    }

    Mat(std::initializer_list<Type> list){
        refCount = new int32_t;
        *refCount = 1;

        ndim = 1;
        dims = new size_type[ndim];
        dims[0] = list.size();
        buildStrides();

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

        ndim = sizeof...(arg);
        errorCheck(ndim > 32, "too many dimensions (<=32 allowed)");

        dims = new size_type[ndim]{(static_cast<size_type>(ind))...};
        errorCheck(list.size() != size(),
            "Initializer list size inconsistent with dimensions");
        buildStrides();
     
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
        ndim = b.ndim;
        dims = new size_type[ndim];
        for(long i = 0; i < ndim; i++){
            dims[i] = b.dims[i];
        }
        strides = new size_type[ndim];
        for(long i = 0; i < ndim; i++){
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
        for(long i = 0; i < ndim; i++){
            offset += temp[i]*strides[i];
        }
        return data[offset];
    }

    template<typename... arg>
    const Type& operator() (const arg... ind) const{
        size_type temp[sizeof...(arg)] = {(static_cast<size_type>(ind))...};
        size_type offset = 0;
        for(long i = 0; i < ndim; i++){
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
        ndim = b.ndim;
        dims = new size_type[ndim];
        for(long i = 0; i < ndim; i++){
            dims[i] = b.dims[i];
        }
        strides = new size_type[ndim];
        for(long i = 0; i < ndim; i++){
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
        return broadcast(b, And<Type,bool>);
    }

    template<class Type2>
    Mat<bool> operator|(const Mat<Type2> &b){
        return broadcast(b, Or<Type,Type2>);
    }

    Mat<bool> operator|(bool b){
        return broadcast(b, Or<Type,bool>);
    }

    Mat<bool> operator!(); // defined below

    Mat<bool> operator==(const Mat<Type> b){
        return broadcast(b, Equality<Type>);
    }

    Mat<bool> operator==(Type b){
        return broadcast(b, Equality<Type>);
    }

    Mat<bool> operator!=(const Mat<Type> b){
        return broadcast(b, Inequality<Type>);
    }

    Mat<bool> operator!=(Type b){
        return broadcast(b, Inequality<Type>);
    }

    Mat<bool> operator<(const Mat<Type> b){
        return broadcast(b, LessThan<Type>);
    }

    Mat<bool> operator<(Type b){
        return broadcast(b, LessThan<Type>);
    }

    Mat<bool> operator<=(const Mat<Type> b){
        return broadcast(b, LessThanEqual<Type>);
    }

    Mat<bool> operator<=(Type b){
        return broadcast(b, LessThanEqual<Type>);
    }

    Mat<bool> operator>(const Mat<Type> b){
        return broadcast(b, GreaterThan<Type>);
    }

    Mat<bool> operator>(Type b){
        return broadcast(b, GreaterThan<Type>);
    }

    Mat<bool> operator>=(const Mat<Type> b){
        return broadcast(b, GreaterThanEqual<Type>);
    }

    Mat<bool> operator>=(Type b){
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

        size_type* big_dim;
        size_type* small_dim;
        long dimdiff;
        if(ndim >= b.ndim){
            big_dim = dims;
            small_dim = b.dims;
            dimdiff = ndim - b.ndim;
            out.ndim = ndim;
            out.dims = new size_type[ndim];
            out.strides = new size_type[ndim];
        }
        else{
            big_dim = b.dims;
            small_dim = dims;
            dimdiff = b.ndim - ndim;
            out.ndim = b.ndim;
            out.dims = new size_type[b.ndim];
            out.strides = new size_type[b.ndim];
        }

        for(long i = 0; i < dimdiff; i++){
            out.dims[i] = big_dim[i];
        }
        for(long i = dimdiff; i < out.ndim; i++){
            errorCheck(big_dim[i] != small_dim[i - dimdiff] &&
                        big_dim[i] != 1 && small_dim[i - dimdiff] != 1,
                        "operand frames not aligned");
            if(big_dim[i] >= small_dim[i - dimdiff]) out.dims[i] = big_dim[i];
            else out.dims[i] = small_dim[i - dimdiff];
        }
        out.buildStrides();

        delete[] out.memory;
        out.memory = new Type3[out.size()];
        out.data = out.memory;

        broadcast(b, f, out);
        return out;
    }

    template<class Type2, class Type3>
    void broadcast(const Mat<Type2> &b, Type3 (*f)(Type, Type2), Mat<Type3> &out){
        size_type effstrideA[32], effstrideB[32];
        if(ndim >= b.ndim) broadcastHelper(*this, b, out, effstrideA, effstrideB);
        else broadcastHelper(b, *this, out, effstrideB, effstrideA);

        size_type posA = 0, posB = 0, posOut = 0;
        size_type coord[32];
        for(long i = 0; i < out.ndim; i++){
            coord[i] = 0;
        }
        for(size_type i = 0; i < out.size(); i++){
            out.data[posOut] = f(this->data[posA],b.data[posB]);
            for(long j = out.ndim-1; j >= 0; j--){
                coord[j]++;
                if(coord[j] != out.dims[j]){
                    posA += effstrideA[j];
                    posB += effstrideB[j];
                    posOut += out.strides[j];
                    break;
                }
                else{
                    coord[j] = 0;
                    posA -= effstrideA[j]*(out.dims[j]-1);
                    posB -= effstrideB[j]*(out.dims[j]-1);
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
        Mat<Type> temp(copy());
        for(auto& i : temp){
            i *= -1;
        }
        return temp;
    }

    Mat operator^ (const Mat<Type> &b){
        errorCheck(ndim != 2 || b.ndim != 2,
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
        errorCheck(sizeof...(arg) > static_cast<size_type>(2*out.ndim),
                    "too many arguments for roi function");
        int temp[sizeof...(arg)] = {(static_cast<int>(ind))...};

        for(long i = 0; i < out.ndim; i++){
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
        errorCheck(memory == dest.memory,
            "Source and destination matrix share same backing data");
        t().copy(dest);
        return dest;
    }

    Mat T(){
        errorCheck(ndim != 2, "hard in-place transpose may only be used on 2d matrix");
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
                "hard in-place transpose may only be used on square or contiguous matrices");
        }
        return *this;
    }

    Mat t() const{
        Mat<Type> dest(*this);
        for(int i = 0, j = ndim-1; i < ndim; i++, j--){
            dest.strides[i] = strides[j];
            dest.dims[i] = dims[j];
        }
        return dest;
    }

    void print(){
        print(stdout);
        return;
    }

    void print(FILE* output){
        iterator i = begin();
        long j;
        long endframe;

        while(i != end()){
            if(i.coord[ndim-1] == 0){
                j = ndim-1;
                for(; j >= 0; j--){
                    if(i.coord[j] != 0){
                        break;
                    }
                }
                for(long k = 0; k < ndim; k++){
                    if(k <= j) fprintf(output, " ");
                    else fprintf(output, "[");
                }
            }

            fprintf(output, "%g", (double)(*i));

            if(i.coord[ndim-1] != dims[ndim-1]-1) fprintf(output, " ");
            else{
                endframe = 0;
                for(j = ndim - 1; j >= 0; j--){
                    if(i.coord[j] == dims[j]-1){
                        fprintf(output, "]");
                        endframe++;
                    }
                    else break;
                }
                if(i.index != size()-1){
                    for(j = 0; j < endframe; j++){
                        fprintf(output, "\n");
                    }
                }
            }
            i++;
        }
        fprintf(output, "\n");
        return;
    }

    template<class newType = Type>
    Mat<newType> copy() const{
        Mat<newType> dest(size());
        dest.ndim = ndim;
        delete[] dest.dims;
        dest.dims = new size_type[ndim];
        dest.strides = new size_type[ndim];
        for(long i = 0; i < ndim; i++){
            dest.dims[i] = dims[i];
        }
        dest.buildStrides();
        copy(dest);
        return dest;
    }

    template<class newType>
    void copy(Mat<newType>& dest) const;

    void scalarFill(Type x){
        for(auto& i : *this){
            i = x;
        }
    }

    template<typename... arg>
    void reshape(const arg... ind){
        errorCheck(!isContiguous(),
            "Cannot reshape non-contiguous matrix");

        long autodim = -1;
        errorCheck(static_cast<long>(sizeof...(arg)) > 32,
            "too many arguments to reshape function");
        errorCheck(sizeof...(arg) == 0, "reshape requires at least one parameter");

        long temp[sizeof...(arg)] = {(static_cast<long>(ind))...};
        size_type shapecheck = 1, autoLength;
        for(long i = 0; i < static_cast<long>(sizeof...(arg)); i++){
            errorCheck(temp[i] < -1, "matrix dimensions can not be negative");
            if(temp[i] == -1){
                errorCheck(autodim != -1, "too many inferred dimensions in reshape");
                autodim = i;
            }
            else shapecheck *= temp[i];
        }
        if(autodim == -1) errorCheck(shapecheck != size(), "new shape size mismatch");
        else{
            errorCheck(size() % shapecheck != 0, "reshape dimension inferrence failed");
            autoLength = size() / shapecheck;
        }
        
        ndim = sizeof...(arg);
        delete[] dims;
        dims = new size_type[ndim];
        for(long i = 0; i < ndim; i++){
            if(i == autodim) dims[i] = autoLength;
            else dims[i] = temp[i];
        }
        buildStrides();
    }

    static Mat<Type> wrap(Type* data, long new_ndim,
                            size_type* new_dims, size_type* strides = NULL){
        Mat<Type> result;
        delete[] result.dims;
        delete[] result.strides;
        delete[] result.memory;
        result.ndim = new_ndim;
        result.dims = new size_type[result.ndim];
        result.strides = new size_type[result.ndim];
        for(long i = 0; i < result.ndim; i++){
            result.dims[i] = new_dims[i];
            if(strides != NULL) result.strides[i] = strides[i];
        }
        if(strides == NULL) result.buildStrides();

        result.customTypeData = data;
        result.memory = data;
        result.data = data;
        return result;
    }

    static Mat<Type> wrap(Type* data, long new_ndim,
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
        result.ndim = new_ndim;
        result.dims = new size_type[result.ndim];
        result.strides = new size_type[result.ndim];
        for(long i = 0; i < result.ndim; i++){
            result.dims[i] = new_dims[i];
            result.strides[i] = new_strides[i];
        }
        result.memory = data;
        result.data = data;
        result.customTypeData = arr;
        result.customDestructor = destructor;
        return result;
    }

    static Mat<Type> wrap(Type* data, long new_ndim,
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
        result.ndim = new_ndim;
        result.dims = new size_type[result.ndim];
        result.strides = new size_type[result.ndim];
        for(long i = 0; i < result.ndim; i++){
            result.dims[i] = new_dims[i];
            result.strides[i] = new_strides[i];
        }
        result.memory = data;
        result.data = data;
        result.customTypeData = arr;
        result.customDestructor = destructor;
        return result;
    }

    template<typename... arg>
    static Mat zeros(const arg... ind){
        Mat result(ind...);
        for(auto& i: result){
            i = 0;
        }
        return result;
    }

    static Mat zeros_like(const Mat a){
        Mat result = empty_like(a);
        for(auto& i: result){
            i = 0;
        }
        return result;
    }

    template<typename... arg>
    static Mat ones(const arg... ind){
        Mat result(ind...);
        for(auto& i: result){
            i = 1;
        }
        return result;
    }

    static Mat ones_like(const Mat a){
        Mat result = empty_like(a);
        for(auto& i: result){
            i = 1;
        }
        return result;
    }

    static Mat<Type> empty_like(const Mat<Type> a){
        Mat<Type> result(a.size());
        result.ndim = a.ndim;
        delete[] result.dims;
        result.dims = new size_type[a.ndim];
        for(long i = 0; i < a.ndim; i++){
            result.dims[i] = a.dims[i];
        }
        result.buildStrides();
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
        Mat<Type> temp({a},1);
        return temp.broadcast(b, f);
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
    size_t coord[32];
    size_t eff_last_dim;

    MatIter(Mat<Type>& mat, size_t ind) : matrix(mat), index(ind){
        mat.errorCheck(ind > mat.size(),
            "iterator index greater than matrix size");
        for(long i = 0; i < matrix.ndim; i++){
            coord[i] = 0;
        }
        for(long i = matrix.ndim-1; i >= 0; i--){
            if(matrix.dims[i] != 1){
                eff_last_dim = i;
                break;
            }
        }
        if(ind == 0) return;
        if(ind == matrix.size()){
            position = matrix.size();
            for(long i = 0; i < matrix.ndim; i++){
                position *= matrix.strides[i];
            }
            return;
        }
        size_t temp = matrix.size(), remainder = index;
        for(long i = 0; i < matrix.ndim; i++){
            temp /= matrix.dims[i];
            coord[i] = remainder / temp;
            remainder = index % temp;
        }
    }
    
    bool operator==(MatIter b){
        matrix.errorCheck(matrix.data != b.matrix.data,
            "Comparison between iterators of different matrices");
        return index == b.index;
    }

    bool operator!=(MatIter b){
        matrix.errorCheck(matrix.data != b.matrix.data,
            "Comparison between iterators of different matrices");
        return index != b.index;
    }

    MatIter& operator++(){
        index++;
        for(int i = eff_last_dim; i >= 0; i--){
            coord[i]++;
            if(coord[i] != matrix.dims[i]){
                position += matrix.strides[i];
                break;
            }
            else{
                coord[i] = 0;
                position -= matrix.strides[i]*(matrix.dims[i]-1);
            }
        }
        return *this;
    }

    MatIter operator++(int){
        MatIter<Type> clone(*this);
        ++(*this);
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
    size_t coord[32];
    size_t eff_last_dim;

    Const_MatIter(const Mat<Type>& mat, size_t ind) : matrix(mat){
        for(int i = 0; i < matrix.ndim; i++){
            coord[i] = 0;
        }
        for(long i = matrix.ndim-1; i >= 0; i--){
            if(matrix.dims[i] != 1){
                eff_last_dim = i;
                break;
            }
        }
        if(ind == 0) return;
        if(ind == matrix.size()){
            index = ind;
            position = matrix.size();
            for(int i = 0; i < matrix.ndim; i++){
                position *= matrix.strides[i];
            }
        }
        while(index < ind){
            ++(*this);
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
        for(int i = eff_last_dim; i >= 0; i--){
            coord[i]++;
            if(coord[i] != matrix.dims[i]){
                position += matrix.strides[i];
                break;
            }
            else{
                coord[i] = 0;
                position -= matrix.strides[i]*(matrix.dims[i]-1);
            }
        }
        return *this;
    }

    Const_MatIter operator++(int){
        Const_MatIter<Type> clone(*this);
        ++(*this);
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
    for(long i = 0; i < ndim; i++){
        errorCheck(mask.dims[i] != dims[i],
            "mask index broadcasting not yet implemented\n");
    }
    size_type newSize = 0;
    for(auto i : mask){
        if(i) newSize++;
    }
    Mat<Type> out(newSize);
    ito(mask, out);
    return out;
}

template<class Type>
template<class Type2>
Mat<Type> Mat<Type>::i(Mat<Type2> &indices,
                typename std::enable_if<std::is_integral<Type2>::value>::type*){
    errorCheck(indices.ndim != 1, "index lists with ndim != 1 not yet implemented");

    Mat<Type> out(indices.size() * (size() / dims[0]));
    out.ndim = ndim;
    delete[] out.dims;
    out.dims = new size_type[ndim];
    out.dims[0] = indices.size();
    for(int i = 1; i < ndim; i++){
        out.dims[i] = dims[i];
    }
    out.buildStrides();

    ito(indices, out);
    return out;
}

template<class Type>
void Mat<Type>::ito(Mat<bool> &mask, Mat<Type> &out){
    errorCheck(out.ndim != 1,
                "output of ito() function must be 1-dimensional");

    size_type newSize = 0;
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
    errorCheck(indices.ndim != 1,
        "index lists with ndim != 1 not yet implemented");
    errorCheck(out.ndim != ndim,
        "inconsistent number of dimensions in output matrix in call to ito()");
    errorCheck(out.dims[0] != indices.size(),
        "output matrix shape does not match given index list in call to ito()");
    for(long i = 1; i < ndim; i++){
        errorCheck(out.dims[i] != dims[i],
        "output matrix shape mismatch in call to ito()");
    }

    iterator dimend = begin();
    iterator i = begin(); //for iterating the current matrix
    iterator k = out.begin();

    size_type offset = size() / dims[0];
    for(size_type j = 0; j < indices.size();j++){
        i.index = indices(j) * offset;
        i.position = indices(j) * strides[0];
        dimend.index = (indices(j) + 1) * offset;
        for(; i!= dimend; ++i){
            *k = *i;
            ++k;
        }
    }
}

template<class Type>
template<class newType>
void Mat<Type>::copy(Mat<newType>& dest) const{
    errorCheck(dest.ndim != ndim,
        "Matrix dimension mismatch during copy");
    for(long i = 0; i > dest.ndim; i++){
        errorCheck(dest.dims[i] != dims[i],
            "output matrix shape mismatch in call to copy()");
    }
    MatIter<newType> j = dest.begin();
    for(auto i : *this){
        *j = static_cast<newType>(i);
        j++;
    }
    return;
}
