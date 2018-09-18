#ifndef MENOH_ARRAY_HPP
#define MENOH_ARRAY_HPP

#include <algorithm>
#include <memory>
#include <vector>

#include <menoh/dims.hpp>
#include <menoh/dtype.hpp>
#include <menoh/exception.hpp>

namespace menoh_impl {

    class array_profile {
    public:
        array_profile() = default;

        array_profile(dtype_t dtype, std::vector<int> const& dims)
          : dtype_(dtype), dims_(dims) {}

        dtype_t dtype() const { return dtype_; }
        auto const& dims() const { return dims_; }

    private:
        dtype_t dtype_ = dtype_t::undefined;
        std::vector<int> dims_;
    };

    class array {
    public:
        array() = default;

        array(dtype_t d, std::vector<int> const& dims, void* data_handle);

        array(dtype_t d, std::vector<int> const& dims,
              std::shared_ptr<void> data);

        array(dtype_t d, std::vector<int> const& dims);

        array(array_profile const& profile, void* data_handle)
          : array(profile.dtype(), profile.dims(), data_handle) {}

        array(array_profile const& profile, std::shared_ptr<void> const& data)
          : array(profile.dtype(), profile.dims(), data) {}

        explicit array(array_profile const& profile)
          : array(profile.dtype(), profile.dims()) {}

        dtype_t dtype() const { return dtype_; }
        auto const& dims() const { return dims_; }

        auto* data() const { return data_handle_; }
        bool has_ownership() const { return static_cast<bool>(data_); }

    private:
        dtype_t dtype_ = dtype_t::undefined;
        std::vector<int> dims_;

        std::shared_ptr<void> data_;
        void* data_handle_ = nullptr;
    };

    std::size_t total_size(array const& a);

    float* fbegin(array const& a);
    float* fend(array const& a);

    float& fat(array const& a, std::size_t i);

    template <typename T>
    auto uniforms(dtype_t d, std::vector<int> const& dims, T val) {
        static_assert(std::is_arithmetic<T>::value, "");
        auto arr = array(d, dims);
        if(d == dtype_t::float_) {
            std::fill_n(static_cast<float*>(arr.data()), calc_total_size(dims),
                        static_cast<float>(val));
            return arr;
        }
        throw invalid_dtype(std::to_string(static_cast<int>(d)));
    }

    array zeros(dtype_t d, std::vector<int> const& dims);

} // namespace menoh_impl

#endif // MENOH_ARRAY_HPP
