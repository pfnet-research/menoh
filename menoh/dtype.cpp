#include <menoh/dtype.hpp>

namespace menoh_impl {

    std::string dtype_to_string(dtype_t dtype) {
        if(dtype == dtype_t::undefined) {
            return "undefined";
        } else if(dtype == dtype_t::float_) {
            return "float";
        }
        throw invalid_dtype(std::to_string(static_cast<int>(dtype)));
    }

    std::size_t get_size_in_bytes(dtype_t d) {
        if(d == dtype_t::float16) {
            return size_in_bytes<dtype_t::float16>;
        } else
        if(d == dtype_t::float32) {
            return size_in_bytes<dtype_t::float32>;
        }else
        if(d == dtype_t::float64) {
            return size_in_bytes<dtype_t::float64>;
        } else
        if(d == dtype_t::int8) {
            return size_in_bytes<dtype_t::int8>;
        } else
        if(d == dtype_t::int16) {
            return size_in_bytes<dtype_t::int16>;
        } else
        if(d == dtype_t::int32) {
            return size_in_bytes<dtype_t::int32>;
        } else
        if(d == dtype_t::int64) {
            return size_in_bytes<dtype_t::int64>;
        }
        throw invalid_dtype(std::to_string(static_cast<int>(d)));
    }

} // namespace menoh_impl
