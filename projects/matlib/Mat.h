#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
using namespace std;

template <class Type>
Type Add(Type a, Type b){ return a + b; };

template <class Type>
Type Subtract(Type a, Type b){ return a - b; };

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
            fprintf(stderr, "%s", message);
            exit(1);
        }
        return;
    }
    public:
        iterator begin(){
            return iterator(*this, 0);
        }
        iterator end(){
            return iterator(*this, strides[0]*columns() + strides[1]*rows());
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
            return this->dims[0];
        }
        size_type columns() const{
            return this->dims[1];
        }
        Mat(size_type a = 1, size_type b = 1){ // make this variadic!
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
            errorCheck(*refCount < 0,"Reference counter is negative somehow\n");
            if(*refCount == 0){
                delete refCount;
                delete []memory;
            }
            delete []dims;
            delete []strides;
        }
        Type& operator() (size_type a, size_type b){
            return data[a*strides[1] + b*strides[0]];
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
                errorCheck(dims[n] != 1 && dims[n] != b.dims[n] && b.dims[n] != 1, "frames are not aligned\n");
                if(dims[n] == 1) x[n] = b.dims[n];
                else x[n] = dims[n];
            }
            Mat<Type> result(x[0], x[1]); //variadic
            delete []x;

            for(size_type i = 0; i < result.size(); i++){
                result(i/result.columns(),i%result.columns()) = (*f)(operator()(i/result.columns()%rows(), i%columns())
                                    , b(i/result.columns()%b.rows(), i%b.columns()));
            }
            return result;
        }
        Mat operator- (){
            Mat<Type> result(rows(), columns());
            copy(result);
            for(auto& i : result){
                i *= -1;
            }
            return result;
        }
        Mat operator^ (const Mat &b){
            if(columns() != b.rows()){
                fprintf(stderr,"Matrix size mismatch\n");
                exit(1);
            }
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
        Mat roi(int rowStart = -1, int rowEnd = -1, int colStart = -1, int colEnd = -1){
            errorCheck(rowStart < -1 || rowStart > static_cast<int>(columns()), "roi argument 1 invalid");
            errorCheck(rowEnd < -1 || rowEnd > static_cast<int>(columns()), "roi argument 2 invalid");
            errorCheck(colStart < -1 || colStart > static_cast<int>(rows()), "roi argument 3 invalid");
            errorCheck(colEnd < -1 || colEnd > static_cast<int>(rows()), "roi argument 4 invalid");

            if(rowStart == -1) rowStart = 0;
            if(rowEnd == -1) rowEnd = static_cast<int>(columns());
            if(colStart == -1) colStart = 0;
            if(colEnd == -1) colEnd = static_cast<int>(rows());
            errorCheck(rowStart == rowEnd || colStart == colEnd, "roi dim cannot equal 0");

            Mat<Type> result(*this);
            result.dims[0] = colEnd-colStart;
            result.dims[1] = rowEnd-rowStart;
            result.data = &memory[colStart*columns() + rowStart];
            return result;
        }
        void T(Mat& dest){
            errorCheck(ndims != 2, "transpose may only be used on 2d matrix");
            errorCheck(dims[0] != dest.dims[1] || dims[1] != dims[0], "Matrix size mismatch\n");
            errorCheck(memory == dest.memory, "Source and destination matrix share same backing data\n");
            errorCheck(data == dest.data, "TODO: call other transpose function\n");
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
                Mat<Type> dest(rows(),columns()); //variadic
                size_t n = 0;
                for(auto i : *this){
                    dest.data[n] = i;
                    n++;
                }
                return dest;
        }
        void copy(Mat<Type>& dest){
            errorCheck(dest.ndims != ndims, "Matrix dimension mismatch");
            for(size_type i = 0; i > dest.ndims; i++){
                errorCheck(dest.dims[i] != dims[i], "Matrix size mismatch");
            }
            size_t n = 0;
            for(auto i : *this){
                dest(0,n) = i;
                n++;
            }
            return;
        }
        bool inbounds(size_type a, size_type b){
            if(a >= 0 && a < columns() && b >= 0 && b < rows()) return true;
            else return false;
        }
        bool isContiguous(){
            //zero dimensional?
            if(strides[0] != 1) return false;
            for(int i = 1; i < ndims; i++){
                strides[i] != dims[i];
            }
        }
        void scalarFill(Type x){
            for(size_type n = 0; n < rows();n++){
                for(size_type i = 0; i < columns();i++){
                    operator()(n,i) = x;
                }
            }
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
            if((position-offset)%matrix.strides[1] == 0 && position >= offset){
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
            if((position-offset)%matrix.strides[1] == 0 && position >= offset){
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
