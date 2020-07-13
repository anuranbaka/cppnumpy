#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
using namespace std;

template <class Type>
inline Type Add(Type a, Type b){ return a + b; };

template <class Type>
inline Type Subtract(Type a, Type b){ return a - b; };

template <class Type>
class MatIter;

template <class Type = double>
class Mat {
    friend class MatIter<Type>;

    typedef MatIter<Type> iterator;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;
    typedef Type value_type;
    typedef Type * pointer;
    typedef Type & reference;

    size_type ndims = 2;
    size_type* dims;
    size_type* strides;
    Type* memory; 
    Type* data;
    int64_t* refCount;

    void errorCheck(bool e, const char* message) const{
        if(e){
            fprintf(stderr, "%s\n", message);
            exit(1);
        }
        return;
    }
    public:
        iterator begin(){
            return iterator(*this, 0);
        }
        iterator end(){
            if(ndims == 2){
                return iterator(*this, strides[0]*columns() + strides[1]*rows());
            }
            else{
                return iterator(*this, strides[0]*columns());
            }
        }
        size_type size() const{
            if(ndims == 0) return 0;
            size_type result = dims[0];
            for(int i = 1; i < ndims; i++){
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
        bool inbounds(size_type a){
            if(a >= 0 && a < columns()) return true;
            else return false;
        }
        bool inbounds(size_type a, size_type b){
            if(a >= 0 && a < columns() && b >= 0 && b < rows()) return true;
            else return false;
        }
        bool isContiguous(){
            if(strides[0] != 1) return false;
            if(ndims == 2 && strides[1] != dims[1]) return false;
            else if(ndims > 2) errorCheck(true, "n-dimensional contiguity check not yet implemented.");
            return true;
        }
        Mat(size_type a = 1){
            refCount = new int64_t;
            *refCount = 1;
            ndims = 1;
            dims = new size_type[ndims];
            dims[0] = a;
            strides = new size_type[ndims];
            strides[0] = 1;
            memory = new Type[a];
            data = memory;
        }
        Mat(size_type a, size_type b){
            refCount = new int64_t;
            *refCount = 1;
            dims = new size_type[ndims];
            dims[0] = a;
            dims[1] = b;
            strides = new size_type[ndims];
            strides[0] = 1;
            strides[1] = b;
            memory = new Type[a*b];
            data = memory;
        }
        Mat(std::initializer_list<Type> list, size_type a){
            refCount = new int64_t;
            *refCount = 1;
            ndims = 1;
            dims = new size_type[ndims];
            dims[0] = a;
            errorCheck(list.size() != a, "Initializer list size inconsistent with dimensions");
            strides = new size_type[ndims];
            strides[0] = 1;
            memory = new Type[a];
            data = memory;
            size_type i = 0;
            for(auto elem : list){
                data[i] = elem;
                i++;
            }
        }
        Mat(std::initializer_list<Type> list, size_type a, size_type b){
            refCount = new int64_t;
            *refCount = 1;
            dims = new size_type[ndims];
            dims[0] = a;
            dims[1] = b; // make this variadic
            errorCheck(list.size() != a*b, "Initializer list size inconsistent with dimensions");
            strides = new size_type[ndims];
            strides[0] = 1;
            strides[1] = columns();
            memory = new Type[a*b];
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
            for(int i = 0; i < ndims; i++){
                dims[i] = b.dims[i];
            }
            strides = new size_type[ndims];
            for(int i = 0; i < ndims; i++){
                strides[i] = b.strides[i];
            }
            memory = b.memory;
            data = b.data;
        }
        ~Mat(){
            (*refCount)--;
            errorCheck(*refCount < 0,"Reference counter is negative somehow");
            if(*refCount == 0){
                delete refCount;
                delete []memory;
            }
            delete []dims;
            delete []strides;
        }
        Type& operator() (size_type a){
            return data[a*strides[0]];
        }
        Type& operator() (size_type a, size_type b){
            return data[a*strides[1] + b*strides[0]];
        }
        const Type& operator() (size_type a) const{
            return data[a*strides[0]];
        }
        const Type& operator() (size_type a, size_type b) const{
            return data[a*strides[1] + b*strides[0]];
        }
        Type& operator() (iterator i){
            return data[i.position];
        }
        Mat& operator= (const Mat &b){
            (*refCount)--;
            errorCheck(*refCount < 0,"Reference counter is negative somehow\n");
            if(*refCount == 0){
                delete refCount;
                delete[] memory;
            }
            delete[] dims;
            delete[] strides;
            refCount = b.refCount;
            (*refCount)++;
            ndims = b.ndims;
            dims = new size_type[ndims];
            for(int i = 0; i < ndims; i++){
                dims[i] = b.dims[i];
            }
            strides = new size_type[ndims];
            for(int i = 0; i < ndims; i++){
                strides[i] = b.strides[i];
            }
            memory = b.memory;
            data = b.data;
            return *this;
        }
        Mat& operator= (const Type scalar){
            for(auto& i: *this){
                i = scalar;
            }
            return *this;
        }
        Mat operator+(const Mat &b){
            return broadcast(b, Add);
        }
        Mat operator- (const Mat &b){
            return broadcast(b, Subtract);
        }
        Mat broadcast(const Mat &b, Type (*f)(Type, Type)){
            size_type* x;
            if(ndims >= b.ndims) x = new size_type[ndims];
            else x = new size_type[b.ndims];

            for(size_type n = 0; n < ndims; n++){
                errorCheck(dims[n] != 1 && dims[n] != b.dims[n] && b.dims[n] != 1, "frames are not aligned");
                if(dims[n] == 1) x[n] = b.dims[n];
                else x[n] = dims[n];
            }
            Mat<Type> result;
            if(ndims == 2){
                Mat<Type> temp(x[0], x[1]);
                result = temp;
                for(size_type i = 0; i < result.size(); i++){
                    result(i/result.columns(),i%result.columns()) = (*f)(operator()(i/result.columns()%rows(), i%columns())
                                        , b(i/result.columns()%b.rows(), i%b.columns()));
                }
            }
            else{
                Mat<Type> temp(x[0]);
                result = temp;
                for(size_type i = 0; i < result.size(); i++){
                    result(i) = (*f)(operator()(i%columns()) , b(i%b.columns()));
                }
            }
            delete []x;

            return result;
        }
        Mat operator- (){
            Mat<Type> result;
            if(ndims == 2){
                Mat<Type> temp(rows(), columns());
                result = temp;
            }
            else{
                Mat<Type> temp(columns());
                result = temp;
            }
            copy(result);
            for(auto& i : result){
                i *= -1;
            }
            return result;
        }
        Mat operator^ (const Mat &b){
            errorCheck(ndims != 2 || b.ndims != 2, "Matrix multiply only available on 2d matrices");
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
        Mat roi(int dim1Start = -1, int dim1End = -1, int dim2Start = -1, int dim2End = -1){
            if(ndims == 1){
                errorCheck(ndims == 1 && (dim2Start != -1 || dim2End != -1), "Too many arguments for 1d matrix");
                errorCheck(dim1Start < -1 || dim1Start > static_cast<int>(columns()), "roi argument 1 invalid");
                errorCheck(dim1End < -1 || dim1End > static_cast<int>(columns()), "roi argument 2 invalid");
                errorCheck(dim2Start != -1, "Too many arguments");
                errorCheck(dim2End != -1, "Too many arguments");
            }
            else if(ndims == 2){
                errorCheck(dim1Start < -1 || dim1Start > static_cast<int>(rows()), "roi argument 1 invalid");
                errorCheck(dim1End < -1 || dim1End > static_cast<int>(rows()), "roi argument 2 invalid");
                errorCheck(dim2Start < -1 || dim2Start > static_cast<int>(columns()), "roi argument 3 invalid");
                errorCheck(dim2End < -1 || dim2End > static_cast<int>(columns()), "roi argument 4 invalid");
            }

            if(dim1Start == -1) dim1Start = 0;
            if(dim1End == -1 && ndims == 2) dim1End = static_cast<int>(rows());
            else if(dim1End == -1 && ndims == 1) dim1End = static_cast<int>(columns());
            if(dim2Start == -1 && ndims == 2) dim2Start = 0;
            if(dim2End == -1 && ndims == 2) dim2End = static_cast<int>(columns());

            Mat<Type> result(*this);
            result.dims[0] = dim1End-dim1Start;
            if(ndims == 2){
                result.dims[1] = dim2End-dim2Start;
                result.data = &memory[dim1Start*columns() + dim2Start];
            }
            else
            {
                result.data = &memory[dim1Start];
            }
            return result;
        }
        void T(Mat& dest){
            errorCheck(ndims != 2, "transpose may only be used on 2d matrix");
            errorCheck(dims[0] != dest.dims[1] || dims[1] != dims[0], "Matrix size mismatch");
            errorCheck(memory == dest.memory, "Source and destination matrix share same backing data");
            errorCheck(data == dest.data, "TODO: call other transpose function");
            if(isContiguous()){
                for(size_type i=0;i<size();i++){
                    dest(0,i) = operator()(i%rows(),i/rows());
                }
            }
            else{
                for(size_type i=0; i<rows(); i++){
                    for(size_type j=0; j<columns(); j++){
                        dest(i,j) = operator()(j,i);
                    }
                }
            }
            return;
        }
        Mat T(){
            errorCheck(ndims != 2, "transpose may only be used on 2d matrix");
            Mat<Type> dest(columns(),rows());
            if(isContiguous()){
                for(size_type i=0;i<size();i++){
                    dest(0,i) = operator()(i%rows(),i/rows());
                }
            }
            else{
                for(size_type i=0; i<rows(); i++){
                    for(size_type j=0; j<columns(); j++){
                        dest(i,j) = operator()(j,i);
                    }
                }
            }
            return dest;
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
        Mat copy(){
            Mat<Type> dest;
            if(ndims == 2){
                Mat<Type> temp(rows(),columns());
                dest = temp;
            }
            else{
                Mat<Type> temp(columns());
                dest = temp;
            }
            size_t n = 0;
            for(auto i : *this){
                dest.data[n] = i;
                n++;
            }
            return dest;
        }
        void copy(Mat<Type>& dest){
            errorCheck(dest.ndims != ndims, "Matrix dimension mismatch during copy");
            for(size_type i = 0; i > dest.ndims; i++){
                errorCheck(dest.dims[i] != dims[i], "Matrix size mismatch");
            }
            size_t m = 0;
            size_t n = 0;
            for(auto i : *this){
                dest(m,n) = i;
                n++;
                if(n == columns()){
                    n = 0;
                    m++;
                }
            }
            return;
        }
        void scalarFill(Type x){
            for(auto& i : *this){
                i = x;
            }
        }
        void reshape(int new_dim1 = -1){
            errorCheck(!isContiguous(), "Cannot reshape non-contiguous matrix");
            errorCheck(new_dim1 < -1, "matrix cannot have negative dimensions");
            if(new_dim1 == -1) new_dim1 = size();
            else errorCheck(size() != new_dim1, "new shape size mismatch");

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
            errorCheck(!isContiguous(), "Cannot reshape non-contiguous matrix");
            errorCheck(new_dim1 < -1 || new_dim2 < -1, "matrix cannot have negative dimensions");
            errorCheck(new_dim1 == -1 && new_dim2 == -1, "only one argument of reshape can be -1");
            if(new_dim1 == -1) new_dim1 = size()/new_dim2;
            else if(new_dim2 == -1) new_dim2 = size()/new_dim1;
            else errorCheck(size() != new_dim1 * new_dim2, "new shape size mismatch");

            if(ndims == 2){
                dims[0] = static_cast<size_type>(new_dim1);
                dims[1] = static_cast<size_type>(new_dim2);
                strides[1] = static_cast<size_type>(new_dim2);
            }
            else{
                ndims = 2;
                delete[] dims;
                delete[] strides;
                dims = new size_type[ndims];
                dims[0] = new_dim1;
                dims[1] = new_dim2;
                strides = new size_type[ndims];
                strides[0] = 1;
                strides[1] = new_dim2;
            }
            return;
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
};

template <class Type>
class MatIter{
    public:
        Mat<Type>& matrix;
        size_t position;
        MatIter(Mat<Type>& mat, size_t pos) : matrix(mat), position(pos){}
        
        bool operator==(MatIter b){
            matrix.errorCheck(matrix.data != b.matrix.data, "Comparison between iterators of different matrices");
            if(position == b.position) return true;
            else return false;
        }
        bool operator!=(MatIter b){
            matrix.errorCheck(matrix.data != b.matrix.data, "Comparison between iterators of different matrices");
            if(position != b.position) return true;
            else return false;
        }
        MatIter& operator++(){
            size_t offset = matrix.strides[0]*(matrix.columns()-1);
            if(matrix.ndims == 1){
                if(position >= matrix.columns()-1) position = matrix.columns();
                else position += matrix.strides[0];
            }
            else if((position-offset)%matrix.strides[1] == 0 && position >= offset){
                if(position >= (matrix.columns()-1)*matrix.strides[0] + (matrix.rows()-1)*matrix.strides[1]){
                    position = matrix.strides[0]*matrix.columns() + matrix.strides[1]*matrix.rows(); //end condition
                }
                else{
                    position -= matrix.strides[0]*(matrix.columns() - 1);
                    position += matrix.strides[1];
                }
            }
            else position += matrix.strides[0];
            return *this;
        }
        MatIter operator++(int){
            MatIter<Type> clone(*this);
            size_t offset = matrix.strides[0]*(matrix.columns()-1);
            if(matrix.ndims == 1){
                if(position >= matrix.columns()-1) position = matrix.columns();
            }
            else if((position-offset)%matrix.strides[1] == 0 && position >= offset){
                if(position >= (matrix.columns()-1)*matrix.strides[0] + (matrix.rows()-1)*matrix.strides[1]){
                    position = matrix.strides[0]*matrix.columns() + matrix.strides[1]*matrix.rows(); //end condition
                }
                else{
                    position -= matrix.strides[0]*(matrix.columns() - 1);
                    position += matrix.strides[1];
                }
            }
            else position += matrix.strides[0];
            return clone;
        }
        Type & operator*(){
            return matrix.data[position];
        }
};