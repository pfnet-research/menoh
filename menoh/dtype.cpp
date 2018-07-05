#include <menoh/dtype.hpp>

namespace menoh_impl {

    std::string dtype_to_string(dtype_t dtype) {
        if(dtype == dtype_t::undefined) {
            return "undefined";
        } 
        else if (dtype == dtype_t::int32) {
            return "int32";
        }
        else if (dtype == dtype_t::int64) {
            return "int64";
        }
        else if(dtype == dtype_t::float_) {
            return "float";
        }
        throw invalid_dtype(std::to_string(static_cast<int32_t>(dtype)));
    }

} // namespace menoh_impl
