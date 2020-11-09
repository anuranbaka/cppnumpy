#include <pybind11/pybind11.h> //pybind11 
#include <matPybind.h> //allows conversion of pyarray->Mat

Mat<double> makeMat(){
    Mat<double> out;
    return out;
}
PYBIND11_MODULE(pyCapsuleTest, m){
    m.def("makeMat", &makeMat, "tests pycapsule");
}