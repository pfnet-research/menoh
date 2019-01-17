#ifndef MENOH_DTYPE_HPP
#define MENOH_DTYPE_HPP

#include <cassert>
#include <climits>
#include <cstdint>
#include <limits>
#include <string>

#include <include/menoh/menoh.h>

#include <menoh/exception.hpp>

namespace menoh_impl {

    static_assert(CHAR_BIT == 8, "Checking one byte is 8bit: failed");

    static_assert(std::numeric_limits<float>::is_iec559,
                  "Checking one float is 32bit: failed");

    enum class dtype_t {
        undefined = -1,
        float_ = menoh_dtype_float,
        float16 = menoh_dtype_float16,
        float32 = menoh_dtype_float32,
        float64 = menoh_dtype_float64,
        int8 = menoh_dtype_int8,
        int16 = menoh_dtype_int16,
        int32 = menoh_dtype_int32,
        int64 = menoh_dtype_int64,
        // TODO more types
    };
    static_assert(dtype_t::undefined != dtype_t::float_, "");

    class invalid_dtype : public exception {
    public:
        invalid_dtype(std::string const& dtype)
          : exception(menoh_error_code_invalid_dtype,
                      "menoh invalid dtype error: " + dtype) {}
    };

    std::string dtype_to_string(dtype_t dtype);

    template <dtype_t>
    struct dtype_to_type {};

    template <>
    struct dtype_to_type<dtype_t::float16> {
        using type = std::int16_t;
    };

    template <>
    struct dtype_to_type<dtype_t::float32> { // including dtype_t::float_
        using type = float;
    };

    template <>
    struct dtype_to_type<dtype_t::float64> {
        using type = double;
    };

    template <>
    struct dtype_to_type<dtype_t::int8> {
        using type = std::int8_t;
    };

    template <>
    struct dtype_to_type<dtype_t::int16> {
        using type = std::int16_t;
    };

    template <>
    struct dtype_to_type<dtype_t::int32> {
        using type = std::int32_t;
    };

    template <>
    struct dtype_to_type<dtype_t::int64> {
        using type = std::int64_t;
    };

    template <dtype_t d>
    using dtype_to_type_t = typename dtype_to_type<d>::type;

    template <dtype_t d>
    constexpr int size_in_bytes = sizeof(dtype_to_type_t<d>);

    inline std::size_t get_size_in_bytes(dtype_t dtype) {
        if(dtype == dtype_t::float_) {
            return size_in_bytes<dtype_t::float_>;
        } else
        if(dtype == dtype_t::float16) {
            return size_in_bytes<dtype_t::float16>;
        } else
        if(dtype == dtype_t::float32) {
            return size_in_bytes<dtype_t::float32>;
        } else
        if(dtype == dtype_t::float64) {
            return size_in_bytes<dtype_t::float64>;
        } else
        if(dtype == dtype_t::int8) {
            return size_in_bytes<dtype_t::int8>;
        } else
        if(dtype == dtype_t::int16) {
            return size_in_bytes<dtype_t::int8>;
        } else
        if(dtype == dtype_t::int32) {
            return size_in_bytes<dtype_t::int8>;
        } else
        if(dtype == dtype_t::int64) {
            return size_in_bytes<dtype_t::int8>;
        }
    }

} // namespace menoh_impl

#endif // MENOH_DTYPE_HPP
