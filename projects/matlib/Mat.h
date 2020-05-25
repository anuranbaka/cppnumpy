#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
using namespace std;

template <class Type = double>
class Mat {
    size_t ndims = 2;
    size_t* dims;
    //size_t* strides;
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
        size_t rows() const{
            return this->dims[0];
        }
        size_t columns() const{
            return this->dims[1];
        }
        size_t size() const{
            return rows()*columns();
        }
        Mat(size_t a, size_t b){
            refCount = new int64_t;
            *refCount = 1;
            data = new Type[a*b];
            dims = new size_t[ndims];
            dims[0] = a;
            dims[1] = b;
            //strides = new size_t[ndims];
            //strides[0] = sizeof(Type);
            //strides[1] = sizeof(Type)*columns();
        }
        Mat(std::initializer_list<Type> list, size_t a, size_t b){
            refCount = new int64_t;
            *refCount = 1;
            dims = new size_t[ndims];
            dims[0] = a;
            dims[1] = b;
            //strides = new size_t[ndims];
            //strides[0] = sizeof(Type);
            //strides[1] = sizeof(Type)*columns();
            data = new Type[a*b];
            size_t i = 0;
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
            dims = b.dims;
            //strides = b.strides;
        }
        ~Mat(){
            (*refCount)--;
            errorCheck(*refCount < 0,"Reference counter is negative somehow\n");
            if(*refCount == 0){
                delete refCount;
                delete []data;
                delete []dims;
                //delete []strides;
            }
        }
        Type& operator() (size_t a, size_t b){
            errorCheck(a < 0 || b < 0 || a + b > size(),"Element access outside matrix scope\n");
            return data[a*columns() + b];
            //return *(data + a*strides[1] + b*strides[0]);
        }
        const Type& operator() (size_t a, size_t b) const{
            errorCheck(a < 0 || b < 0 || a + b > size(),"Element access outside matrix scope\n");
            return data[a*columns() + b];
            //return *(data + a*strides[1] + b*strides[0]);
        }
        Mat& operator= (const Mat &b){
            (*refCount)--;
            errorCheck(*refCount < 0,"Reference counter is negative somehow\n");
            if(*refCount == 0){
                delete refCount;
                delete[] data;
                delete[] dims;
                //delete[] strides;
            }
            refCount = b.refCount;
            (*refCount)++;
            data = b.data;
            dims = b.dims;
            //strides = b.strides;
            return *this;
        }
        Mat operator+ (Mat &b){
            size_t i = 0;
            size_t* x;
            if(ndims >= b.ndims) x = new size_t[ndims];
            else x = new size_t[b.ndims];

            for(size_t n = 0; n < ndims; n++){
                errorCheck(dims[n] != 1 && dims[n] != b.dims[i] && b.dims[i] != 1, "frames are not aligned\n");
                if(dims[n] == 1) x[i] = b.dims[i];
                else x[i] = dims[i];
                i++;
            }
            Mat<Type> result(x[0], x[1]); //update later when higher dimension support
            delete []x;

            size_t width;
            if(columns() > b.columns()) width = columns();
            else width = b.columns();
            for(size_t i = 0; i < result.size(); i++){
                result(0,i) = operator()(i/width%rows(),i%columns()) + b(i/width%b.rows(),i%b.columns());
            }
            return result;
        }
        Mat operator- (Mat &b){
            size_t i = 0;
            size_t* x;
            if(ndims >= b.ndims) x = new size_t[ndims];
            else x = new size_t[b.ndims];

            for(size_t n = 0; n < ndims; n++){
                errorCheck(dims[n] != 1 && dims[n] != b.dims[i] && b.dims[i] != 1, "frames are not aligned\n");
                if(dims[n] == 1) x[i] = b.dims[i];
                else x[i] = dims[i];
                i++;
            }
            Mat<Type> result(x[0], x[1]); //update later when higher dimension support
            delete []x;

            size_t width;
            if(columns() > b.columns()) width = columns();
            else width = b.columns();
            for(size_t i = 0; i < result.size(); i++){
                result(0,i) = operator()(i/width%rows(),i%columns()) - b(i/width%b.rows(),i%b.columns());
            }
            return result;
        }
        Mat operator- (){
            Mat<Type> result(rows(),columns());
            for(size_t i = 0; i < size(); i++){
                result(0,i) = operator()(0,i) * -1;
            }
            return result;
        }
        Mat operator^ (Mat &b){
            if(columns() != b.rows()){
                fprintf(stderr,"Matrix size mismatch\n");
                exit(1);
            }
            Mat<Type> result(rows(),b.columns());
            Type sum;
            for(size_t x = 0; x < rows(); x++){
                for(size_t i = 0; i<b.columns();i++){
                    sum = 0;
                    for(size_t n = 0; n< columns(); n++){
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
            for(size_t i=0;i<size();i++){
                dest(0,i) = operator()(i%rows(),i/rows());
            }
            size_t temp = dest.dims[0];
            if(columns() != dest.rows()) dest.dims[0] = dest.dims[1];
            if(rows() != dest.columns()) dest.dims[1] = temp;
            return;
        }
        Mat T(){
            Mat<Type> dest(columns(),rows());
            for(size_t i=0;i<size();i++){
                dest(0,i) = operator()(i%rows(),i/rows());
            }
            return dest;
        }
        void print(){
            for(size_t i = 0; i<size();i++){
                printf("%g",operator()(0,i));
                if(i%columns() == columns() - 1) printf("\n");
                else printf(", ");
            }
        }
};
