#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <initializer_list>
#include <matMath.h>
#include <type_traits>

using namespace std;
#ifndef ERROR_IF_FAILED_ALLOC
#define ERROR_IF_FAILED_ALLOC false
#endif
const long MAX_NDIM = 32;

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
inline Type BitAnd(Type a, Type b){ return a & b; }

template <class Type>
inline Type BitOr(Type a, Type b){ return a | b; }

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
template <class Type, class Type2>
class iMat;
template <class Type = double>
class Mat;
template <class Type = double>
class MatBase;

struct DimInfo{
    long ndim;
    size_t dims[MAX_NDIM];
    size_t strides[MAX_NDIM];
    DimInfo(long new_ndim, size_t* new_dims, size_t* new_strides) : ndim(new_ndim){
        for(long i = 0; i < ndim; i++){
            dims[i] = new_dims[i];
        }
        for(long i = 0; i < ndim; i++){
            strides[i] = new_strides[i];
        }
    }
};

template<class Type>
class AllocInfo{
    public:
        void* userdata = NULL;
        //allocates dims, strides and (if not copying another matrix) base
        void (*allocateMeta)(Mat<Type>&, void*, long) = NULL;
        void (*deallocateMeta)(Mat<Type>&) = NULL;
        void (*allocateData)(MatBase<Type>&, void*, const size_t) = NULL;
        void (*deallocateData)(MatBase<Type>&) = NULL;
};

template <class Type /*double*/>
class MatBase{
    public:
    int32_t refCount = 0;
    AllocInfo<Type>* allocator = NULL;
    Type* data = NULL;

    MatBase(AllocInfo<Type>* allocIn = NULL) : allocator(allocIn) {}

    MatBase(void* newdata) : data((Type*)newdata) {}
    
    ~MatBase(){
        if(allocator == NULL || allocator->deallocateData == NULL){
            delete[] data;
        }
        else{
            allocator->deallocateData(*this);
        }
    }
};

template <class Type /*double*/>
class Mat {
    friend class MatIter<Type>;
    friend class Const_MatIter<Type>;


    template<class left, class right, class Type3>
    static void broadcastHelper(const Mat<left>& bigMat, const Mat<right>& smallMat,
                                Mat<Type3>& out,
                                size_t* bigStrides, size_t* smallStrides){
        long dimdiff = bigMat.ndim - smallMat.ndim;
        if(out.ndim != bigMat.ndim)
            throw invalid_argument("output matrix ndim not equal to broadcasted ndim");
        for(int i = 0; i < dimdiff; i++){
            if(out.dims[i] != bigMat.dims[i])
                throw invalid_argument("broadcast output matrix shape mismatch");
        }
        for(int i = dimdiff; i < bigMat.ndim; i++){
            if(bigMat.dims[i] != 1 && smallMat.dims[i-dimdiff] != 1){
                if(bigMat.dims[i] != smallMat.dims[i-dimdiff])
                    throw invalid_argument("operand frames not aligned");
                if(out.dims[i] != bigMat.dims[i])
                    throw invalid_argument("broadcast output matrix shape mismatch");
            }
            if(bigMat.dims[i] == 1)
                if(out.dims[i] != smallMat.dims[i-dimdiff])
                    throw invalid_argument("broadcast output matrix shape mismatch");
            if(smallMat.dims[i-dimdiff] == 1)
                if(out.dims[i] != bigMat.dims[i])
                    throw invalid_argument("broadcast output matrix shape mismatch");
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
    
    typedef MatIter<Type> iterator;
    typedef Const_MatIter<Type> const_iterator;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;
    typedef Type value_type;
    typedef Type * pointer;
    typedef Type & reference;

    AllocInfo<Type>* allocator = NULL;
    Type* data = NULL;
    long ndim = 0;
    size_type* dims = NULL;
    size_type* strides = NULL;
    MatBase<Type>* base = NULL;

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
        if(ndim < 1) throw out_of_range("matrix has no dimensions");
        if(ndim == 1) throw out_of_range("1d matrix has no rows");
        return this->dims[ndim - 2];
    }

    size_type columns() const{
        if(ndim < 1) throw out_of_range("matrix has no dimensions");
        return this->dims[ndim - 1];
    }

    template<typename... arg>
    bool inbounds(const arg... ind){
        if(static_cast<long>(sizeof...(arg)) > ndim)
            throw out_of_range("too many coordinates passed to inbounds()");
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

    void buildStrides(){
        if(ndim == 0)
            throw length_error("Cannot build strides for 0 dimensional matrix");
        if(strides == NULL)
            throw runtime_error("Matrix strides cannot be NULL");
        strides[ndim-1] = 1;
        for(long j = ndim-2; j >= 0; j--){
            strides[j] = strides[j+1]*dims[j+1];
        }
    }

    Mat(){
        thread_local MatBase<Type> emptySingleton;
        base = &emptySingleton;
        if(base->refCount == 0) (base->refCount)++;
        (base->refCount)++;
    }

    template<typename... arg>
    explicit Mat(const arg... ind){
        ndim = sizeof...(arg);
        if(ndim > MAX_NDIM) throw invalid_argument
            ("too many dimensions given in Mat constructor");

        dims = new size_type[ndim]{(static_cast<size_type>(ind))...};
        strides = new size_type[ndim];
        buildStrides();

        MatBase<Type>* newBase;
        newBase = new MatBase<Type>();
        base = newBase;
        base->refCount++;

        base->data = new Type[size()];
        data = base->data;
    }

    template<typename... arg>
    Mat(AllocInfo<Type>* alloc, const arg... ind){
        ndim = sizeof...(arg);
        if(ndim > MAX_NDIM) throw invalid_argument
            ("too many dimensions given in Mat constructor");
        allocator = alloc;

        if(allocator == NULL || allocator->allocateMeta == NULL){
            dims = new size_type[ndim];
            strides = new size_type[ndim];

            MatBase<Type>* newBase;
            newBase = new MatBase<Type>();
            base = newBase;
        }
        else allocator->allocateMeta(*this, alloc->userdata, ndim);
        base->refCount++;

        size_type temp_dims[sizeof...(arg)] = {(static_cast<size_type>(ind))...};
        for(long i = 0; i < ndim; ++i){
            dims[i] = temp_dims[i];
        }
        buildStrides();

        if(allocator == NULL || allocator->allocateData == NULL)
            base->data = new Type[size()];
        else allocator->allocateData(*base, alloc->userdata, size());
        data = base->data;
    }

    Mat(std::initializer_list<Type> list){
        ndim = 1;

        dims = new size_type[ndim];
        strides = new size_type[ndim];

        MatBase<Type>* newBase;
        newBase = new MatBase<Type>();
        base = newBase;
        base->refCount++;

        dims[0] = list.size();
        buildStrides();

        base->data = new Type[size()];
        data = newBase->data;

        size_type i = 0;
        for(auto elem : list){
            data[i] = elem;
            i++;
        }
    }

    Mat(std::initializer_list<Type> list, AllocInfo<Type>* alloc){
        ndim = 1;
        allocator = alloc;

        if(allocator->allocateMeta == NULL){
            dims = new size_type[ndim];
            strides = new size_type[ndim];

            MatBase<Type>* newBase;
            newBase = new MatBase<Type>();
            base = newBase;
        }
        else allocator->allocateMeta(*this, alloc->userdata, ndim);
        base->refCount++;

        dims[0] = list.size();
        buildStrides();

        if(allocator->allocateData == NULL)
            base->data = new Type[size()];
        else
            allocator->allocateData(*base, alloc->userdata, size());
        data = base->data;

        size_type i = 0;
        for(auto elem : list){
            data[i] = elem;
            i++;
        }
    }

    template<typename... arg>
    Mat(std::initializer_list<Type> list, const arg... ind){
        ndim = sizeof...(arg);
        if(ndim > MAX_NDIM) throw invalid_argument
            ("too many dimensions given in Mat constructor");

        dims = new size_type[ndim]{(static_cast<size_type>(ind))...};
        if(list.size() != size()) throw invalid_argument
            ("Initializer list size inconsistent with dimensions");
        strides = new size_type[ndim];
        buildStrides();

        MatBase<Type>* newBase;
        newBase = new MatBase<Type>();
        base = newBase;
        base->refCount++;

        base->data = new Type[size()];
        data = newBase->data;

        size_type i = 0;
        for(auto elem : list){
            data[i] = elem;
            i++;
        }
    }

    template<typename... arg>
    Mat(std::initializer_list<Type> list, AllocInfo<Type>* alloc, const arg... ind){
        ndim = sizeof...(arg);
        if(ndim > MAX_NDIM) throw invalid_argument
            ("too many dimensions given in Mat constructor");
        allocator = alloc;

        if(allocator->allocateMeta == NULL){
            dims = new size_type[ndim];
            if(list.size() != size()) throw invalid_argument
                    ("Initializer list size inconsistent with dimensions");
            strides = new size_type[ndim];

            MatBase<Type>* newBase;
            newBase = new MatBase<Type>();
            base = newBase;
        }
        else allocator->allocateMeta(*this, alloc->userdata, ndim);
        base->refCount++;

        size_type temp[sizeof...(arg)] = {(static_cast<size_type>(ind))...};
        for(long i = 0; i < ndim; ++i){
            dims[i] = temp[i];
        }
        if(list.size() != size())
            throw invalid_argument("Initializer list size inconsistent with dimensions");
        buildStrides();

        if(allocator->allocateData == NULL){
            base->data = new Type[size()];
        }
        else
            allocator->allocateData(*base, allocator->userdata, size());
        data = base->data;

        size_type i = 0;
        for(auto elem : list){
            data[i] = elem;
            i++;
        }
    }

    Mat(const Mat& b){
        base = b.base;
        base->refCount++;
        ndim = b.ndim;
        data = b.data;
        allocator = b.allocator;
        
        if(allocator == NULL || allocator->allocateMeta == NULL){
            dims = new size_type[ndim];
            strides = new size_type[ndim];
        }
        else{
            allocator->allocateMeta(*this, allocator->userdata, ndim);
        }

        for(long i = 0; i < ndim; i++){
            dims[i] = b.dims[i];
        }
        for(long i = 0; i < ndim; i++){
            strides[i] = b.strides[i];
        }
    }

    Mat(const iMat<Type, bool>& b){
        size_t newSize = 0;
        for(auto i : b.index){
            if(i) newSize++;
        }
        ndim = 1;
        allocator = b.matrix.allocator;

        if(allocator == NULL || allocator->allocateMeta == NULL){
            dims = new size_type[ndim];
            strides = new size_type[ndim];
            
            MatBase<Type>* newBase;
            newBase = new MatBase<Type>();
            base = newBase;
        }
        else allocator->allocateMeta(*this, allocator->userdata, ndim);
        base->refCount++;

        dims[0] = newSize;
        buildStrides();

        if(allocator == NULL || allocator->allocateData == NULL){
            base->data = new Type[newSize];
        }
        else allocator->allocateData(*base, allocator->userdata, newSize);
        data = base->data;
        
        b.matrix.ito(b.index, *this);
    }

    template<class Type2>
    Mat(const iMat<Type, Type2>& b){
        size_t newSize = b.index.size();
        for(int i = 1; i < b.matrix.ndim; i++){
            newSize *= b.matrix.dims[i];
        }
        ndim = b.matrix.ndim;
        allocator = b.matrix.allocator;

        if(allocator == NULL || allocator->allocateMeta == NULL){
            dims = new size_type[ndim];
            strides = new size_type[ndim];

            MatBase<Type>* newBase;
            newBase = new MatBase<Type>();
            base = newBase;
        }
        else{
            allocator->allocateMeta(*this, allocator->userdata, ndim);
        }
        base->refCount++;

        dims[0] = b.index.size();
        for(int i = 1; i < ndim; i++){
            dims[i] = b.matrix.dims[i];
        }
        buildStrides();

        if(allocator == NULL || allocator->allocateData == NULL){
            base->data = new Type[newSize];
        }
        else allocator->allocateData(*base, allocator->userdata, newSize);
        data = base->data;

        b.matrix.ito(b.index, *this);
    }

    ~Mat(){
        (base->refCount)--;
        if(allocator == NULL || allocator->deallocateMeta == NULL){
            delete []dims;
            delete []strides;
            if(base->refCount <= 0) delete base;
        }
        else allocator->deallocateMeta(*this);
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
        base = b.base;
        (base->refCount)++;
        ndim = b.ndim;
        data = b.data;
        allocator = b.allocator;
        
        if(allocator == NULL || allocator->allocateMeta == NULL){
            dims = new size_type[ndim];
            strides = new size_type[ndim];
        }
        else{
            allocator->allocateMeta(*this, allocator->userdata, ndim);
        }

        for(long i = 0; i < ndim; i++){
            dims[i] = b.dims[i];
        }
        for(long i = 0; i < ndim; i++){
            strides[i] = b.strides[i];
        }
        return *this;
    }

    Mat<Type>& operator= (Type scalar){
        for(auto& i: *this){
            i = scalar;
        }
        return *this;
    }

    Mat<Type>& operator=(const iMat<Type, bool> b){
        this->~Mat<Type>();
        ndim = 1;
        allocator = b.matrix.allocator;

        if(allocator == NULL || allocator->allocateMeta == NULL){
            dims = new size_type[ndim];
            strides = new size_type[ndim];

            MatBase<Type>* newBase;
            newBase = new MatBase<Type>();
            base = newBase;
        }
        else{
            allocator->allocateMeta(*this, allocator->userdata, ndim);
        }
        base->refCount++;
        
        dims[0] = 0;
        for(auto i : b.index){
            if(i) dims[0]++;
        }
        buildStrides();

        if(allocator == NULL || allocator->allocateData == NULL)
            base->data = new Type[size()];
        else
            allocator->allocateData(*base, allocator->userdata, size());
        data = base->data;

        b.matrix.ito(b.index, *this);
        return *this;
    }

    template<class Type2>
    Mat<Type>& operator=(const iMat<Type, Type2> b){
        this->~Mat<Type>();
        ndim = b.matrix.ndim;
        
        if(allocator == NULL){
            dims = new size_type[ndim];
            strides = new size_type[ndim];

            MatBase<Type>* newBase;
            newBase = new MatBase<Type>();
            base = newBase;
        }
        else{
            allocator->allocateMeta(*this, allocator->userdata, ndim);
        }
        base->refCount++;

        dims[0] = b.index.size();
        for(long i = 1; i < ndim; i++){
            dims[i] = b.matrix.dims[i];
        }
        buildStrides();

        if(allocator == NULL || allocator->allocateData == NULL)
            base->data = new Type[size()];
        else
            allocator->allocateData(*base, allocator->userdata, size());
        data = base->data;

        b.matrix.ito(b.index, *this);
        return *this;
    }
    
    Mat<Type> operator+(const Mat<Type> &b){
        return broadcast(b, Add<Type>);
    }

    Mat<Type> operator+(Type b){
        return broadcast(b, Add<Type>);
    }

    void operator +=(const Mat<Type> &b){
        if(base != NULL && b.base != NULL && base == b.base)
            broadcast(b.copy(), Add<Type>, *this);
        else broadcast(b, Add<Type>, *this);
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
        if(base != NULL && b.base != NULL && base == b.base)
            broadcast(b.copy(), Subtract<Type>, *this);
        else broadcast(b, Subtract<Type>, *this);
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
        if(base != NULL && b.base != NULL && base == b.base)
            broadcast(b.copy(), Multiply<Type>, *this);
        else broadcast(b, Multiply<Type>, *this);
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
        if(base != NULL && b.base != NULL && base == b.base)
            broadcast(b.copy(), Divide<Type>, *this);
        else broadcast(b, Divide<Type>, *this);
    }

    void operator /=(Type b){
        broadcast(b, Divide<Type>, *this);
    }

    template<class Type2>
    Mat<bool> operator&&(const Mat<Type2> &b){
        return broadcast(b, And<Type,Type2>);
    }

    Mat<bool> operator&&(bool b){
        return broadcast(b, And<Type,bool>);
    }

    template<class Type2>
    Mat<bool> operator||(const Mat<Type2> &b){
        return broadcast(b, Or<Type,Type2>);
    }

    Mat<bool> operator||(bool b){
        return broadcast(b, Or<Type,bool>);
    }

    Mat<bool> operator!(); // defined below

    Mat<Type> operator&(const Mat<Type> &b){
        return broadcast(b, BitAnd<Type>);
    }

    Mat<Type> operator&(Type b){
        return broadcast(b, BitAnd<Type>);
    }

    Mat<Type> operator|(const Mat<Type> &b){
        return broadcast(b, BitOr<Type>);
    }

    Mat<Type> operator|(Type b){
        return broadcast(b, BitOr<Type>);
    }

    Mat<Type> operator~(); // defined below

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

    //currently does not support custom allocation.
    //If needed, allocate output matrix first and call overload below.
    template<class Type2, class Type3>
    Mat<Type3> broadcast(const Mat<Type2> &b, Type3 (*f)(Type, Type2)){
        Mat<Type3> out;

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
            if(big_dim[i] != small_dim[i - dimdiff] &&
                        big_dim[i] != 1 && small_dim[i - dimdiff] != 1)
                throw invalid_argument("operand frames not aligned");
            if(big_dim[i] >= small_dim[i - dimdiff]) out.dims[i] = big_dim[i];
            else out.dims[i] = small_dim[i - dimdiff];
        }
        out.buildStrides();

        MatBase<Type3>* newBase;
        newBase = new MatBase<Type3>();
        out.base = newBase;
        out.base->refCount++;

        out.base->data = new Type3[out.size()];
        out.data = out.base->data;

        broadcast(b, f, out);
        return out;
    }

    template<class Type2, class Type3>
    void broadcast(const Mat<Type2> &b, Type3 (*f)(Type, Type2), Mat<Type3> &out){
        size_type effstrideA[MAX_NDIM], effstrideB[MAX_NDIM];
        if(ndim >= b.ndim) broadcastHelper(*this, b, out, effstrideA, effstrideB);
        else broadcastHelper(b, *this, out, effstrideB, effstrideA);

        size_type posA = 0, posB = 0, posOut = 0;
        size_type coord[MAX_NDIM];
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
        Mat<Type3> out = empty_like<Type3>(*this);
        broadcast(b, f, out);
        return out;
    }

    template<class Type2, class Type3>
    void broadcast(Type2 b, Type3 (*f)(Type, Type2), Mat<Type3> &out){
        if(out.size() != size())
            throw invalid_argument("broadcast output matrix size mismatch");
        iterator j = begin();
        for(auto& i : out){
            i = f((*j), b);
            j++;
        }
        return;
    }

    Mat operator- (){
        Mat<Type> temp(copy());
        for(auto& i : temp){
            i *= -1;
        }
        return temp;
    }

    Mat operator^ (const Mat<Type> &b){
        if(ndim != 2 || b.ndim != 2)
            throw invalid_argument("Matrix multiply only available on 2d matrices");
        if(columns() != b.rows())
            throw invalid_argument("Matrix size mismatch");
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
        if(sizeof...(arg) > static_cast<size_type>(2*out.ndim))
            throw out_of_range("too many arguments for roi function");
        int temp[sizeof...(arg)] = {(static_cast<int>(ind))...};

        for(long i = 0; i < out.ndim; i++){
            if(static_cast<size_type>(2*i) >= sizeof...(arg)) break;
            else if(static_cast<size_type>(2*i)+1 >= sizeof...(arg)){
                if(temp[2*i] == -1) temp[2*i] = 0;
                if(temp[(2*i)] < 0
                        || static_cast<size_type>(temp[2*i]) > out.dims[i])
                    throw invalid_argument("roi shape mismatch");
                out.dims[i] -= temp[2*i];
                out.data += temp[2*i]*out.strides[i];
            }
            else{
                if(temp[(2*i)+1] == -1) temp[(2*i)+1] = out.dims[i];
                if(temp[2*i] == -1) temp[2*i] = 0;
                if(temp[(2*i)+1] < 0 || temp[(2*i)] < 0
                        || static_cast<size_type>(temp[(2*i)+1]) > out.dims[i]
                        || static_cast<size_type>(temp[2*i]) > out.dims[i])
                    throw invalid_argument("roi shape mismatch");
                out.dims[i] = temp[(2*i)+1] - temp[2*i];
                out.data += temp[2*i]*out.strides[i];
            }
        }
        return out;
    }

    //i has 4 versions depending on whether the given parameter is a boolean
    //mask or a list of indices. The default parameter in the indexed version
    //simply causes substitution to fail when a floating point matrix is passed.
    iMat<Type, bool> i(const Mat<bool> &mask);
    template<typename Type2>
    iMat<Type, Type2> i(const Mat<Type2> &indices,
            typename std::enable_if<std::is_integral<Type2>::value>::type* = 0);
    void ito(const Mat<bool> &mask, Mat<Type> &out) const;
    template<typename Type2>
    void ito(const Mat<Type2> &indices, Mat<Type> &out,
            typename std::enable_if<std::is_integral<Type2>::value>::type* = 0) const;

    Mat T(Mat& dest){
        if(base == dest.base){
            throw invalid_argument
                ("Source and destination matrix share same backing data");
            }
        t().copy(dest);
        return dest;
    }

    Mat T(){
        if(ndim != 2) throw invalid_argument
                ("hard in-place transpose may only be used on 2d matrix");
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
            Mat<Type> clone(copy());
            (*this) = (reshape(columns(), rows()));
            for(size_type i = 0; i < columns(); i++){
                for(size_type j = 0; j < rows(); j++){
                    operator()(j,i) = clone(i,j);
                }
            }
        }
        else{
            throw logic_error("T() may only be used on square or contiguous matrices");
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
        Mat<newType> dest(Mat<Type>::empty_like<newType>(*this));
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
    Mat<Type> reshape(const arg... ind){
        long autodim = -1;
        long temp[sizeof...(arg)] = {(static_cast<long>(ind))...};
        size_type shapecheck = 1, autoLength;

        if(static_cast<long>(sizeof...(arg)) > MAX_NDIM)
            throw invalid_argument("too many arguments to reshape function");
        if(sizeof...(arg) == 0)
            throw invalid_argument("reshape requires at least one parameter");
        for(long i = 0; i < static_cast<long>(sizeof...(arg)); i++){
            if(temp[i] < -1) throw out_of_range("matrix dimensions can not be negative");
            if(temp[i] == -1){
                if(autodim != -1)
                    throw invalid_argument("too many inferred dimensions in reshape");
                autodim = i;
            }
            else shapecheck *= temp[i];
        }
        if(autodim == -1 && shapecheck != size())
            throw invalid_argument("new shape size mismatch");
        else{
            if(size() % shapecheck != 0)
                throw invalid_argument("reshape dimension inferrence failed");
            autoLength = size() / shapecheck;
        }

        Mat<Type> out;
        if(isContiguous()){
            out = *this;
        }
        else{
            out = copy();
        }
        out.ndim = sizeof...(arg);
        out.allocator = allocator;

        if(out.allocator == NULL || out.allocator->deallocateMeta == NULL){
            delete[] out.dims;
            delete[] out.strides;
        }
        else out.allocator->deallocateMeta(*this);

        if(out.allocator == NULL || out.allocator->allocateMeta == NULL){
            out.dims = new size_type[out.ndim];
            out.strides = new size_type[out.ndim];
        }
        else{
            out.allocator->allocateMeta(out, out.allocator->userdata, out.ndim);
        }

        for(long i = 0; i < out.ndim; i++){
            if(i == autodim) out.dims[i] = autoLength;
            else out.dims[i] = temp[i];
        }
        out.buildStrides();

        return out;
    }

    static Mat<Type> arange(int stop){
        if(stop < 0) throw out_of_range("arange stop must be >= 0");
        Mat<Type> out(stop);
        int j = 0;
        for(auto& i : out){
            i = j;
            j++;
        }
        return out;
    }

    static Mat<Type> arange(int start, int stop, int step = 1, AllocInfo<Type>* alloc = NULL){
        if(start < 0) throw out_of_range("arange start must be >= 0");
        if(stop < 0) throw out_of_range("arange stop must be >= 0");
        if(step == 0) throw runtime_error("attempted division by zero in arange()");
        size_type newSize = 0;
        if(start <= stop && step > 0) newSize = 1 + ((stop - start - 1) / step);
        else if(stop < start && step < 0) newSize = 1 + ((start - stop - 1) / -step);
        Mat<Type> out(alloc, newSize);
        int j = start;
        for(auto& i : out){
            i = j;
            j += step;
        }
        return out;
    }

    static Mat<Type> wrap(Type* data, long new_ndim,
                            size_type* new_dims, size_type* strides = NULL){
        if(new_ndim < 0) throw out_of_range("number of dimensions cannot be negative");
        if(new_ndim == 0) throw out_of_range("0 dimensional matrices not implemented");
        if(new_ndim > MAX_NDIM) throw out_of_range("wrapped matrix has too many dimensions");
        Mat<Type> result;
        result.ndim = new_ndim;
        result.dims = new size_type[result.ndim];
        result.strides = new size_type[result.ndim];
        for(long i = 0; i < result.ndim; i++){
            result.dims[i] = new_dims[i];
            if(strides != NULL) result.strides[i] = strides[i];
        }
        if(strides == NULL) result.buildStrides();

        MatBase<Type>* newBase;
        newBase = new MatBase<Type>(data);
        result.base = newBase;
        result.base->refCount++;
        result.data = data;

        thread_local AllocInfo<Type> wrap_alloc;
        wrap_alloc.deallocateData = [](MatBase<Type>&){return;};

        result.allocator = &wrap_alloc;
        result.base->allocator = &wrap_alloc;

        return result;
    }

    static Mat<Type> wrap(Type* data, long new_ndim,
                        size_type* new_dims, size_type* new_strides, 
                        AllocInfo<Type>* alloc){
        if(new_ndim < 0) throw out_of_range("number of dimensions cannot be negative");
        if(new_ndim == 0) throw out_of_range("0 dimensional matrices not implemented");
        if(new_ndim > MAX_NDIM) throw out_of_range("wrapped matrix has too many dimensions");
        Mat<Type> result;
        result.allocator = alloc;
        result.ndim = new_ndim;

        if(alloc == NULL || result.allocator->allocateMeta == NULL){
            result.dims = new size_type[result.ndim];
            result.strides = new size_type[result.ndim];

            MatBase<Type>* newBase;
            newBase = new MatBase<Type>();
            result.base = newBase;
        }
        else{
            result.allocator->allocateMeta(result, result.allocator->userdata, result.ndim);
        }
        result.base->allocator = alloc;
        result.base->refCount++;

        for(long i = 0; i < result.ndim; i++){
            result.dims[i] = new_dims[i];
            result.strides[i] = new_strides[i];
        }

        result.base->data = data;
        result.data = result.base->data;

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

    template<class newType = Type>
    static Mat<newType> empty_like(const Mat<Type> a){
        AllocInfo<newType>* alloc = NULL;
        if(a.allocator != NULL){
            if(is_same<Type,newType>()){
                alloc = (AllocInfo<newType>*)a.allocator;
            }
            else{
                static_assert(!ERROR_IF_FAILED_ALLOC,
                        "failed to inherit allocator during type conversion");
            }
        }
        Mat<newType> result(alloc, a.size());

        result.ndim = a.ndim;
        if(result.allocator != NULL){
            if((result.allocator->allocateMeta == NULL
                && result.allocator->deallocateMeta != NULL)
                || (result.allocator->allocateMeta != NULL
                && result.allocator->deallocateMeta == NULL))
                    throw runtime_error("inherited allocator must define both allocation and deallocation procedures");
        }
        if(result.allocator == NULL || result.allocator->deallocateMeta == NULL){
            delete[] result.dims;
            delete[] result.strides;
        }
        else result.allocator->deallocateMeta(result);

        if(result.allocator == NULL || result.allocator->allocateMeta == NULL){
            result.dims = new size_type[result.ndim];
            result.strides = new size_type[result.ndim];
        }
        else result.allocator->allocateMeta(result, result.allocator->userdata, result.ndim);

        result.base->data = result.data;

        for(long i = 0; i < result.ndim; i++){
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
        Mat<Type3> out = empty_like(b);
        iterator j = b.begin();
        for(auto& i : out){
            i = f(a, (*j));
            j++;
        }
        return out;
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
    size_t coord[MAX_NDIM];
    size_t eff_last_dim;

    MatIter(Mat<Type>& mat, size_t ind) : matrix(mat), index(ind){
        if(ind > mat.size())
            throw out_of_range("iterator index greater than matrix size");
        for(long i = 0; i < matrix.ndim; i++){
            coord[i] = 0;
        }
        eff_last_dim = matrix.ndim-1;
        for(long i = eff_last_dim; i >= 0; i--){
            if(matrix.dims[i] != 1){
                eff_last_dim = i;
                break;
            }
        }
        if(ind == 0) return;
        if(ind == matrix.size()){
            position = matrix.size();
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
        if(matrix.data != b.matrix.data)
            throw invalid_argument("Comparison between iterators of different matrices");
        return index == b.index;
    }

    bool operator!=(MatIter b){
        if(matrix.data != b.matrix.data)
            throw invalid_argument("Comparison between iterators of different matrices");
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
    size_t coord[MAX_NDIM];
    size_t eff_last_dim;

    Const_MatIter(const Mat<Type>& mat, size_t ind) : matrix(mat){
        for(int i = 0; i < matrix.ndim; i++){
            coord[i] = 0;
        }
        eff_last_dim = matrix.ndim-1;
        for(long i = eff_last_dim; i >= 0; i--){
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
        if(matrix.data != b.matrix.data)
            throw invalid_argument("Comparison between iterators of different matrices");
        if(index == b.index) return true;
        else return false;
    }

    bool operator!=(Const_MatIter b){
        if(matrix.data != b.matrix.data)
            throw invalid_argument("Comparison between iterators of different matrices");
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

template<class Type, class Type2>
class iMat{
    public:
    Mat<Type> matrix;
    Mat<Type2> index;
    
    iMat(Mat<Type>& m, const Mat<Type2>& i) : matrix(m), index(i){
        if(std::is_same<Type2, bool>::value){
            for(long i = 0; i < matrix.ndim; i++){
                if(index.dims[i] != matrix.dims[i])
                    throw invalid_argument("mask index broadcasting not yet implemented");
            }
        }
        else if(index.ndim != 1)
            throw invalid_argument("index lists with ndim != 1 not yet implemented");
    }

    iMat<Type, Type2>& operator=(const Mat<Type>& b){
        if(matrix.base == b.base){
            return *this = b.copy();
        }
        MatIter<Type> j = matrix.begin();
        Const_MatIter<Type> k = b.begin();
        if(std::is_same<Type2, bool>::value){
            if(b.ndim > 1) throw invalid_argument
                    ("assignment to indexed array requires right operand to be 1d");
            for(auto i : index){
                if(i){
                    if(k == b.end()) throw invalid_argument
                        ("indexed matrix assignment size mismatch");
                    *j = *k;
                    ++k;
                }
                ++j;
            }
            if(k != b.end()) throw invalid_argument
                ("indexed matrix assignment size mismatch");
        }
        else{
            if(b.dims[0] != index.dims[0]) throw invalid_argument
                ("broadcasting assignment to an indexed matrix not yet implemented");
            for(long i = 1; i < matrix.ndim; i++){
                if(b.dims[i] != matrix.dims[i]) throw invalid_argument
                    ("broadcasting assignment to an indexed matrix not yet implemented");
            }
            MatIter<Type> dimend = matrix.begin();
            size_t offset = matrix.size() / matrix.dims[0];

            for(auto i : index){
                j.index = i * offset;
                j.position = i * matrix.strides[0];
                dimend.index = (i + 1) * offset;
                for(; j!= dimend; ++j){
                    *j = *k;
                    ++k;
                }
            }
        };
        return *this;
    }

    Mat<Type>& operator=(Type scalar){
        MatIter<Type> j = matrix.begin();
        if(std::is_same<Type2, bool>::value){
            for(auto i : index){
                if(i) *j = scalar;
                ++j;
            }
        }
        else{
            MatIter<Type> dimend = matrix.begin();
            size_t offset = matrix.size() / matrix.dims[0];

            for(auto i : index){
                j.index = i * offset;
                j.position = i * matrix.strides[0];
                dimend.index = (i + 1) * offset;
                for(; j!= dimend; ++j){
                    *j = scalar;
                }
            }
        };
        return matrix;
    }

    Mat<Type> operator+(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, Add<Type>);
    }
    
    Mat<Type> operator+(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, Add<Type>);
    }

    void operator+=(Mat<Type> b){
        if(matrix.base == b || b.base == *this || matrix.base == b.base)
            broadcast(b.copy(), Add<Type>, *this);
        else *this = *this + b;
    }

    void operator+=(Type b){
        *this = *this + b;
    }

    Mat<Type> operator-(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, Subtract<Type>);
    }

    Mat<Type> operator-(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, Subtract<Type>);
    }

    Mat<Type> operator-(){
        Mat<Type> out(*this);
        return -out;
    }

    void operator-=(Mat<Type> b){
        if(matrix.base == b || b.base == *this || matrix.base == b.base)
            broadcast(b.copy(), Subtract<Type>, *this);
        else *this = *this - b;
    }

    void operator-=(Type b){
        *this = *this - b;
    }

    Mat<Type> operator*(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, Multiply<Type>);
    }
    
    Mat<Type> operator*(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, Multiply<Type>);
    }

    void operator*=(Mat<Type> b){
        if(matrix.base == b || b.base == *this || matrix.base == b.base)
            broadcast(b.copy(), Multiply<Type>, *this);
        else *this = *this * b;
    }

    void operator*=(Type b){
        *this = *this * b;
    }

    Mat<Type> operator/(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, Divide<Type>);
    }
    
    Mat<Type> operator/(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, Divide<Type>);
    }

    void operator/=(Mat<Type> b){
        if(matrix.base == b || b.base == *this || matrix.base == b.base)
            broadcast(b.copy(), Divide<Type>, *this);
        else *this = *this / b;
    }

    void operator/=(Type b){
        *this = *this / b;
    }

    Mat<bool> operator&&(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, And<Type>);
    }
    
    Mat<bool> operator&&(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, And<Type>);
    }

    Mat<bool> operator||(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, Or<Type>);
    }
    
    Mat<bool> operator||(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, Or<Type>);
    }

    Mat<bool> operator!(){
        Mat<Type> out(*this);
        return !out;
    }

    Mat<Type> operator&(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, BitAnd<Type>);
    }
    
    Mat<Type> operator&(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, BitAnd<Type>);
    }

    Mat<Type> operator|(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, BitOr<Type>);
    }
    
    Mat<Type> operator|(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, BitOr<Type>);
    }

    Mat<Type> operator~(){
        Mat<Type> out(*this);
        return ~out;
    }

    Mat<bool> operator==(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, Equality<Type>);
    }
    
    Mat<bool> operator!=(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, Inequality<Type>);
    }

    Mat<bool> operator<(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, LessThan<Type>);
    }
    
    Mat<bool> operator<(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, LessThan<Type>);
    }

    Mat<bool> operator<=(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, LessThanEqual<Type>);
    }
    
    Mat<bool> operator<=(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, LessThanEqual<Type>);
    }

    Mat<bool> operator>(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, GreaterThan<Type>);
    }
    
    Mat<bool> operator>(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, GreaterThan<Type>);
    }

    Mat<bool> operator>=(Mat<Type> b){
        Mat<Type> out(*this);
        return out.broadcast(b, GreaterThanEqual<Type>);
    }
    
    Mat<bool> operator>=(Type b){
        Mat<Type> out(*this);
        return out.broadcast(b, GreaterThanEqual<Type>);
    }

    void print(){
        Mat<Type> out(*this);
        out.print();
    }

    void print(FILE* output){
        Mat<Type> out(*this);
        out.print(output);
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
Mat<bool> operator&&(bool a, Mat<Type> &b){
    return b.broadcast(a, And<Type,bool>);
}

template<class Type>
Mat<bool> operator||(bool a, Mat<Type> &b){
    return b.broadcast(a, Or<Type,bool>);
}

template<class Type>
Mat<Type> operator&(Type a, Mat<Type> &b){
    return b.broadcast(a, BitAnd<Type>);
}

template<class Type>
Mat<Type> operator|(Type a, Mat<Type> &b){
    return b.broadcast(a, BitOr<Type>);
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
Mat<Type> Mat<Type>::operator~(){
    Mat<Type> result(this->copy());
    for(Mat<Type>::iterator i = result.begin(); i != result.end(); ++i){
        *i = ~(*i);
    }
    return result;
}

template<class Type>
Mat<bool> Mat<Type>::operator!(){
    Mat<bool> result(this->copy<bool>());
    for(Mat<bool>::iterator i = result.begin(); i != result.end(); ++i){
        *i = !(*i);
    }
    return result;
}

template<class Type, class Type2>
Mat<Type> operator+(const Type a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, Add<Type>);
}

template<class Type, class Type2>
Mat<Type> operator-(const Type a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, Subtract<Type>);
}

template<class Type, class Type2>
Mat<Type> operator*(const Type a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, Multiply<Type>);
}

template<class Type, class Type2>
Mat<Type> operator/(const Type a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, Divide<Type>);
}

template<class Type, class Type2>
Mat<bool> operator&&(const bool a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, And<Type>);
}

template<class Type, class Type2>
Mat<bool> operator||(const bool a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, Or<Type>);
}

template<class Type, class Type2>
Mat<Type> operator&(const Type a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, BitAnd<Type>);
}

template<class Type, class Type2>
Mat<Type> operator|(const Type a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, BitOr<Type>);
}

template<class Type, class Type2>
Mat<bool> operator==(const Type a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, Equality<Type>);
}

template<class Type, class Type2>
Mat<bool> operator!=(const Type a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, Inequality<Type>);
}

template<class Type, class Type2>
Mat<bool> operator<(const Type a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, LessThan<Type>);
}

template<class Type, class Type2>
Mat<bool> operator<=(const Type a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, LessThanEqual<Type>);
}

template<class Type, class Type2>
Mat<bool> operator>(const Type a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, GreaterThan<Type>);
}

template<class Type, class Type2>
Mat<bool> operator>=(const Type a, const iMat<Type, Type2> &b){
    Mat<Type> out(b);
    return Mat<Type>::broadcast(a, out, GreaterThanEqual<Type>);
}

template<class Type>
iMat<Type, bool> Mat<Type>::i(const Mat<bool> &mask){
    iMat<Type, bool> out(*this, mask);
    return out;
}

template<class Type>
template<class Type2>
iMat<Type, Type2> Mat<Type>::i(const Mat<Type2> &indices,
                typename std::enable_if<std::is_integral<Type2>::value>::type*){
    iMat<Type, Type2> out(*this, indices);
    return out;
}

template<class Type>
void Mat<Type>::ito(const Mat<bool> &mask, Mat<Type> &out)  const{
    if(out.ndim != 1)
        throw invalid_argument("output of ito() function must be 1-dimensional");

    size_type newSize = 0;
    for(auto i : mask){
        if(i) newSize++;
    }
    if(out.size() < newSize)
        throw invalid_argument("insufficient space in output matrix");
    if(out.size() > newSize){
        out = out.roi(0,newSize);
    }

    Mat<bool>::const_iterator j = mask.begin();
    iterator k = out.begin();
    for(const_iterator i = begin(); i != end(); ++i, ++j){
        if(*j){
            *k = *i;
            ++k;
        }
    }
    return;
}

template<class Type>
template<class Type2>
void Mat<Type>::ito(const Mat<Type2> &indices, Mat<Type> &out,
                typename std::enable_if<std::is_integral<Type2>::value>::type*) const{
    if(indices.ndim != 1)
        throw invalid_argument
            ("index lists with ndim != 1 not yet implemented");
    if(out.ndim != ndim)
        throw invalid_argument
            ("inconsistent number of dimensions in output matrix in call to ito()");
    if(out.dims[0] != indices.size())
        throw invalid_argument
            ("output matrix shape does not match given index list in call to ito()");
    for(long i = 1; i < ndim; i++){
        if(out.dims[i] != dims[i])
        throw invalid_argument
            ("output matrix shape mismatch in call to ito()");
    }

    const_iterator dimend = begin();
    const_iterator i = begin(); //for iterating the current matrix
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
    if(dest.ndim != ndim)
        throw invalid_argument("Matrix dimension mismatch during copy");
    for(long i = 0; i > dest.ndim; i++){
        if(dest.dims[i] != dims[i])
            throw invalid_argument("output matrix shape mismatch in call to copy()");
    }
    MatIter<newType> j = dest.begin();
    for(auto i : *this){
        *j = static_cast<newType>(i);
        j++;
    }
    return;
}
