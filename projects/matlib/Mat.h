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
        size_type rows() const{
            return this->dims[0];
        }
        size_type columns() const{
            return this->dims[1];
        }
        size_type size() const{
            return rows()*columns();
        }
        Mat(size_type a = 1, size_type b = 1){
            refCount = new int64_t;
            *refCount = 1;
            memory = new Type[a*b];
            data = memory;
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
            memory = new Type[a*b];
            data = memory;
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
            memory = b.memory;
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
                delete []memory;
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
            size_type n = 0;
            for(auto i : *this){
                result(0,n) = i * -1;
                n++;
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
        Mat copy(){
                Mat<Type> dest(rows(),columns());
                size_t n = 0;
                for(auto i : *this){
                    dest(0,n) = i;
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
