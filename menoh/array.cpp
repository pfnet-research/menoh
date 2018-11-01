#include <menoh/array.hpp>

#include <cstdint>

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
        if(d == dtype_t::float_ || d == dtype_t::float32) {
            // libc++ workaround
            // Below 2 lines are equal to `return std::unique_ptr<float[]>(new
            // float[total_size]);`
            auto u = std::make_unique<float[]>(total_size);
            return std::shared_ptr<void>(u.release(), u.get_deleter());
        } else if(d == dtype_t::int8) {
            auto u = std::make_unique<std::int8_t[]>(total_size);
            return std::shared_ptr<void>(u.release(), u.get_deleter());
        } else if(d == dtype_t::int16) {
            auto u = std::make_unique<std::int16_t[]>(total_size);
            return std::shared_ptr<void>(u.release(), u.get_deleter());
        } else if(d == dtype_t::int32) {
            auto u = std::make_unique<std::int32_t[]>(total_size);
            return std::shared_ptr<void>(u.release(), u.get_deleter());
        } else if(d == dtype_t::int64) {
            auto u = std::make_unique<std::int64_t[]>(total_size);
            return std::shared_ptr<void>(u.release(), u.get_deleter());
        }

        throw invalid_dtype(std::to_string(static_cast<int>(d)));
    }

    array::array(dtype_t d, std::vector<int> const& dims)
      : array(d, dims, allocate_data(d, dims)) {}

    std::size_t total_size(array const& a) { return calc_total_size(a.dims()); }

    dtype_to_type_t<dtype_t::float32>* fbegin(array const& a) {
        return begin<dtype_t::float32>(a);
    }
    dtype_to_type_t<dtype_t::float32>* fend(array const& a) {
        return end<dtype_t::float32>(a);
    }

    dtype_to_type_t<dtype_t::float32>& fat(array const& a, std::size_t i) {
        return at<dtype_t::float32>(a, i);
    }

    array zeros(dtype_t d, std::vector<int> const& dims) {
        return uniforms(d, dims, 0.);
    }

} // namespace menoh_impl
