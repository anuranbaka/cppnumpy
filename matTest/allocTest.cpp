#include <Mat.h>

class mem_buffer{
    void* mem;
    size_t remaining_space;

    public:

    mem_buffer(void* mem, size_t space) : mem(mem), remaining_space(space){};

    template<typename T>
    T* reserve(long size){
        T* out = (T*)mem;
        if(size*sizeof(T) > remaining_space){
            throw bad_alloc();
        }
        mem += size*sizeof(T);
        remaining_space -= size*sizeof(T);
        return out;
    }
};

template<typename T>
void customAllocator_Meta(Mat<T> &mat, void* buf, long ndim){
    mat.dims = ((mem_buffer*)buf)->reserve<size_t>(ndim);
    mat.strides = ((mem_buffer*)buf)->reserve<size_t>(ndim);
    if(mat.base == NULL)
    {
        mat.base = ((mem_buffer*)buf)->reserve<MatBase<T>>(1);
    }
}
template<typename T>
void customDeallocator_Meta(Mat<T>&){
    //maybe write this later
}
template<typename T>
void customAllocator_Data(MatBase<T> &base, void* buf, const size_t size){
    base.data = ((mem_buffer*)buf)->reserve<T>(size);
}
template<typename T>
void customDeallocator_Data(MatBase<T>&){
    //maybe write this later
}
void test(){
    
}

int main (){
    const size_t buff_size = 200000;
    char buffer[buff_size];
    mem_buffer buff(buffer, buff_size);
    void* userdata = &buff;

    AllocInfo<double> alloc_double;
    AllocInfo<size_t> alloc_size_t;
    alloc_double.userdata = userdata;
    alloc_double.allocateMeta = customAllocator_Meta<double>;
    alloc_double.deallocateMeta = customDeallocator_Meta<double>;
    alloc_double.allocateData = customAllocator_Data<double>;
    alloc_double.deallocateData = customDeallocator_Data<double>;
    alloc_size_t.userdata = userdata;
    alloc_size_t.allocateMeta = customAllocator_Meta<size_t>;
    alloc_size_t.deallocateMeta = customDeallocator_Meta<size_t>;
    alloc_size_t.allocateData = customAllocator_Data<size_t>;
    alloc_size_t.deallocateData = customDeallocator_Data<size_t>;
    
    Mat<double> a({0,1,2,3,4,5,6,7,8,9,10,11}, &alloc_double, 3, 4);
    Mat<double> b(&alloc_double, 3, 3);
    Mat<double> c({2,3,4}, &alloc_double);
    Mat<double> d(a);
    Mat<double> e = a.copy();
    Mat<double> f = a.reshape(6,2); // one alloc while this line exists... :/
    Mat<double> g = a.i(Mat<size_t>::arange(2,4,1,&alloc_size_t));

    return 0;
}