#include <menoh/array.hpp>

namespace menoh_impl {

    array::array(dtype_t d, std::vector<int> const& dims, void* data_handle)
      : dtype_(d), dims_(dims), data_(nullptr), data_handle_(data_handle) {}

    array::array(dtype_t d, std::vector<int> const& dims,
                 std::shared_ptr<void> data)
      : dtype_(d), dims_(dims), data_(std::move(data)),
        data_handle_(data_.get()) {}

    std::shared_ptr<void> allocate_data(dtype_t d,
                                        std::vector<int> const& dims) {
        auto total_size = calc_total_size(dims);
        if(d == dtype_t::float_) {
            // libc++ workaround
            // Below 2 lines are equal to `return std::unique_ptr<float[]>(new
            // float[total_size]);`
            auto u = std::make_unique<float[]>(total_size);
            return std::shared_ptr<void>(u.release(), u.get_deleter());
        }
        throw invalid_dtype(std::to_string(static_cast<int>(d)));
    }

    array::array(dtype_t d, std::vector<int> const& dims)
      : array(d, dims, allocate_data(d, dims)) {}

    std::size_t total_size(array const& a) { return calc_total_size(a.dims()); }

    float* fbegin(array const& a) {
        assert(a.dtype() == dtype_t::float_);
        return static_cast<float*>(a.data());
    }
    float* fend(array const& a) {
        assert(a.dtype() == dtype_t::float_);
        return fbegin(a) + total_size(a);
    }

    float& fat(array const& a, std::size_t i) {
        assert(a.dtype() == dtype_t::float_);
        return *(static_cast<float*>(a.data()) + i);
    }

    array zeros(dtype_t d, std::vector<int> const& dims) {
        return uniforms(d, dims, 0.);
    }

} // namespace menoh_impl
