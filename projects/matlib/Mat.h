#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
using namespace std;

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
            return iterator(*this, size());
        }
        size_type rows() const{
            return this->dims[0];
        }
        size_type columns() const{
            return this->dims[1];
        }
        size_type size() const{
            return rows()*columns();
        }
        Mat(size_type a, size_type b){
            refCount = new int64_t;
            *refCount = 1;
            data = new Type[a*b];
            dims = new size_type[ndims];
            dims[0] = a;
            dims[1] = b;
            strides = new size_type[ndims];
            strides[0] = 1;
            strides[1] = columns();
        }
        Mat(std::initializer_list<Type> list, size_type a, size_type b){
            refCount = new int64_t;
            *refCount = 1;
            dims = new size_type[ndims];
            dims[0] = a;
            dims[1] = b;
            strides = new size_type[ndims];
            strides[0] = 1;
            strides[1] = columns();
            data = new Type[a*b];
            size_type i = 0;
            for(auto elem : list){
                errorCheck(i >= size(), "Initializer out of bounds\n");
                data[i] = elem;
                i++;
            }
        }
        Mat(const Mat& b){
            refCount = b.refCount;
            (*refCount)++;
            data = b.data;
            dims = new size_type[ndims];
            dims[0] = b.dims[0];
            dims[1] = b.dims[1];
            strides = new size_type[ndims];
            strides[0] = b.strides[0];
            strides[1] = b.strides[1];
        }
        ~Mat(){
            (*refCount)--;
            errorCheck(*refCount < 0,"Reference counter is negative somehow\n");
            if(*refCount == 0){
                delete refCount;
                delete []data;
            }
            delete []dims;
            delete []strides;
        }
        Type& operator() (size_type a, size_type b){
            errorCheck(a < 0 || b < 0 || a + b > size(),"Element access outside matrix scope\n");
            return data[a*strides[1] + b*strides[0]];
        }
        const Type& operator() (size_type a, size_type b) const{
            errorCheck(a < 0 || b < 0 || a + b > size(),"Element access outside matrix scope\n");
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
                delete[] data;
            }
            delete[] dims;
            delete[] strides;
            refCount = b.refCount;
            (*refCount)++;
            data = b.data;
            dims = new size_type[ndims];
            dims[0] = b.dims[0];
            dims[1] = b.dims[1];
            strides = new size_type[ndims];
            strides[0] = b.strides[0];
            strides[1] = b.strides[1];
            return *this;
        }
        Mat operator+ (const Mat &b){
            size_type i = 0;
            size_type* x;
            if(ndims >= b.ndims) x = new size_type[ndims];
            else x = new size_type[b.ndims];

            for(size_type n = 0; n < ndims; n++){
                errorCheck(dims[n] != 1 && dims[n] != b.dims[i] && b.dims[i] != 1, "frames are not aligned\n");
                if(dims[n] == 1) x[i] = b.dims[i];
                else x[i] = dims[i];
                i++;
            }
            Mat<Type> result(x[0], x[1]);
            delete []x;

            size_type width;
            if(columns() > b.columns()) width = columns();
            else width = b.columns();
            for(size_type i = 0; i < result.size(); i++){
                result(0,i) = operator()(i/width%rows(),i%columns()) + b(i/width%b.rows(),i%b.columns());
            }
            return result;
        }
        Mat operator- (const Mat &b){
            size_type i = 0;
            size_type* x;
            if(ndims >= b.ndims) x = new size_type[ndims];
            else x = new size_type[b.ndims];

            for(size_type n = 0; n < ndims; n++){
                errorCheck(dims[n] != 1 && dims[n] != b.dims[i] && b.dims[i] != 1, "frames are not aligned\n");
                if(dims[n] == 1) x[i] = b.dims[i];
                else x[i] = dims[i];
                i++;
            }
            Mat<Type> result(x[0], x[1]);
            delete []x;

            size_type width;
            if(columns() > b.columns()) width = columns();
            else width = b.columns();
            for(size_type i = 0; i < result.size(); i++){
                result(0,i) = operator()(i/width%rows(),i%columns()) - b(i/width%b.rows(),i%b.columns());
            }
            return result;
        }
        Mat operator- (){
            Mat<Type> result(rows(),columns());
            for(auto i : *this){ //change this to iterator
                result(0,i) = i * -1;
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
                    for(size_type n = 0; n< columns(); n++){
                        sum += operator()(x,n)*b(n,i);
                    }
                    result(x,i) = sum;
                }
            }
            return result;
        }
        void T(Mat& dest){
            errorCheck(size() != dest.size(),"Matrix size mismatch\n");
            errorCheck(data == dest.data, "Source and destination matrix share same backing data\n");
            for(size_type i=0;i<size();i++){
                dest(0,i) = operator()(i%rows(),i/rows());
            }
            size_type temp = dest.dims[0];
            if(columns() != dest.rows()) dest.dims[0] = dest.dims[1];
            if(rows() != dest.columns()) dest.dims[1] = temp;
            return;
        }
        Mat T(){
            Mat<Type> dest(columns(),rows());
            for(size_type i=0;i<size();i++){
                dest(0,i) = operator()(i%rows(),i/rows());
            }
            return dest;
        }
        Mat t() const{
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
            if(position%matrix.strides[1] == matrix.strides[1] - 1 && position >= matrix.strides[0]*(matrix.columns() - 1)){
                if(position == matrix.size() - 1) position = matrix.size();
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
            if(position%matrix.strides[1] == matrix.strides[1] - 1 && position >= matrix.strides[0]*(matrix.columns() - 1)){
                if(position == matrix.size() - 1) position = matrix.size();
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
