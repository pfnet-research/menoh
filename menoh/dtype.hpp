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
        undefined,
        float_ = menoh_dtype_float
        // TODO more types
    };

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
    struct dtype_to_type<dtype_t::float_> {
        using type = float;
    };

    template <dtype_t d>
    using dtype_to_type_t = typename dtype_to_type<d>::type;

    /*
    template <dtype_t d>
    constexpr int size_in_bytes = sizeof(dtype_to_type_t<d>);
    */

} // namespace menoh_impl

#endif // MENOH_DTYPE_HPP
