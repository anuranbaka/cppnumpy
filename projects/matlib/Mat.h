#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
using namespace std;

template <class Type = double>
class Mat {
    size_t ndims = 2;
    size_t* dims;
    Type* data;
    
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
            data = new Type[a*b];
            dims = new size_t[ndims];
            dims[0] = a;
            dims[1] = b;
        }
        Mat(std::initializer_list<Type> list, size_t a, size_t b){
            dims = new size_t[ndims];
            dims[0] = a;
            dims[1] = b;
            data = new Type[a*b];
            size_t i = 0;
            for(auto elem : list){
                this->errorCheck(i >= this->size(), "Initializer out of bounds");
                this->data[i] = elem;
                i++;
            }
        }
        Mat(const Mat& b){
            data = new Type[b.size()];
            dims = new size_t[b.ndims];
            dims[0] = b.rows();
            dims[1] = b.columns();
            for(size_t i = 0; i < this->size(); i++){
                this->operator()(0,i) = b(0,i);
            }

        }
        ~Mat(){
            delete []data;
            delete []dims;
        }
        Type& operator() (size_t a, size_t b){
            this->errorCheck(a < 0 || b < 0 || a + b > this->size(),"Element access outside matrix scope");
            return data[a*columns() + b];
        }
        const Type& operator() (size_t a, size_t b) const{
            this->errorCheck(a < 0 || b < 0 || a + b > this->size(),"Element access outside matrix scope");
            return data[a*columns() + b];
        }
        Mat operator= (const Mat b){
            this->errorCheck(b.size() != this->size(),"Matrix size mismatch during assignment");
            for(size_t i = 0; i < this->size(); i++){
                this->operator()(0,i) = b(0,i);
            }
            return *this;
        }
        Mat operator+ (Mat b){
            size_t i = 0;
            size_t* x;
            if(this->ndims >= b.ndims) x = new size_t[this->ndims];
            else x = new size_t[b.ndims];

            for(size_t n = 0; n < this->ndims; n++){
                this->errorCheck(this->dims[n] != 1 && this->dims[n] != b.dims[i] && b.dims[i] != 1, "frames are not aligned");
                if(this->dims[n] == 1) x[i] = b.dims[i];
                else x[i] = this->dims[i];
                i++;
            }
            Mat<Type> result(x[0], x[1]); //update later when higher dimension support
            delete []x;

            size_t width;
            if(this->columns() > b.columns()) width = this->columns();
            else width = b.columns();
            for(size_t i = 0; i < result.size(); i++){
                result(0,i) = this->operator()(i/width%this->rows(),i%this->columns()) + b(i/width%b.rows(),i%b.columns());
            }
            return result;
        }
        Mat operator- (Mat b){
            size_t i = 0;
            size_t* x;
            if(this->ndims >= b.ndims) x = new size_t[this->ndims];
            else x = new size_t[b.ndims];

            for(size_t n = 0; n < this->ndims; n++){
                this->errorCheck(this->dims[n] != 1 && this->dims[n] != b.dims[i] && b.dims[i] != 1, "frames are not aligned");
                if(this->dims[n] == 1) x[i] = b.dims[i];
                else x[i] = this->dims[i];
                i++;
            }
            Mat<Type> result(x[0], x[1]); //update later when higher dimension support
            delete []x;

            size_t width;
            if(this->columns() > b.columns()) width = this->columns();
            else width = b.columns();
            for(size_t i = 0; i < result.size(); i++){
                result(0,i) = this->operator()(i/width%this->rows(),i%this->columns()) - b(i/width%b.rows(),i%b.columns());
            }
            return result;
        }
        Mat operator- (){
            Mat<Type> result(this->rows(),this->columns());
            for(size_t i = 0; i < this->size(); i++){
                result(0,i) = this->operator()(0,i) * -1;
            }
            return result;
        }
        Mat operator^ (Mat b){
            if(this->columns() != b.rows()){
                fprintf(stderr,"Matrix size mismatch");
                exit(1);
            }
            Mat<Type> result(this->rows(),b.columns());
            Type sum;
            for(size_t x = 0; x < this->rows(); x++){
                for(size_t i = 0; i<b.columns();i++){
                    sum = 0;
                    for(size_t n = 0; n< this->columns(); n++){
                        sum += this->operator()(x,n)*b(n,i);
                    }
                    result(x,i) = sum;
                }
            }
            return result;
        }
        void T(Mat& dest){
            this->errorCheck(this->size() != dest.size(),"Matrix size mismatch");
            this->errorCheck(this->data == dest.data, "Source and destination matrix share same backing data");
            for(size_t i=0;i<this->size();i++){
                dest(0,i) = this->operator()(i%this->rows(),i/this->rows());
            }
            size_t temp = dest.dims[0];
            if(this->columns() != dest.rows()) dest.dims[0] = dest.dims[1];
            if(this->rows() != dest.columns()) dest.dims[1] = temp;
            return;
        }
        Mat T(){
            Mat<Type> dest(this->columns(),this->rows());
            for(size_t i=0;i<this->size();i++){
                dest(0,i) = this->operator()(i%this->rows(),i/this->rows());
            }
            return dest;
        }
        void print(){
            for(size_t i = 0; i<this->size();i++){
                printf("%g",this->operator()(0,i));
                if(i%this->columns() == this->columns() - 1) printf("\n");
                else printf(", ");
            }
        }
};
