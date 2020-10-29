#include <pybind11/pybind11.h>
#include <Mat.h>
#include <Matmodule.h>

int init_numpy(){
     import_array1(0); // PyError if not successful
     return 1;
}

const static int numpy_initialized =  init_numpy();

namespace pybind11 { namespace detail {
    template <class T>
    struct type_caster<Mat<T>> {
    public:
        PYBIND11_TYPE_CASTER(Mat<T>, _("Mat<T>"));

        bool load(handle src, bool) {
            value = wrap_numpy<T>(src.ptr());
            return !PyErr_Occurred();
        }

        static handle cast(Mat<T> src, return_value_policy /* policy */, handle /* parent */) {
            handle result(wrap_mat<T>(src));
            return result;
        }
    };
}}