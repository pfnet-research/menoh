#include <menoh/mkldnn/utility.hpp>

#include <cassert>

#include <menoh/array.hpp>
#include <menoh/dtype.hpp>
#include <menoh/exception.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        std::vector<int32_t> extract_dims(mkldnn::memory const& m) {
            auto const& d = m.get_primitive_desc().desc().data;
            return std::vector<int32_t>(d.dims, d.dims + d.ndims);
        }

        mkldnn::memory::data_type
        dtype_to_mkldnn_memory_data_type(dtype_t dtype) {
            if(dtype == dtype_t::int32) {
                return mkldnn::memory::data_type::s32;
            } else if(dtype == dtype_t::int64) {
                return mkldnn::memory::data_type::s32; // INFO: s64 is not
                                                       // available
            } else if(dtype == dtype_t::float_) {
                return mkldnn::memory::data_type::f32;
            }
            throw invalid_dtype(std::to_string(static_cast<int32_t>(dtype)));
        }

        mkldnn::memory array_to_memory(array const& arr,
                                       std::vector<int32_t> const& dims,
                                       mkldnn::memory::format format,
                                       mkldnn::engine const& engine) {
            // FIXME: mkl-dnn doesn't have s64, so copy
            if(arr.dtype() == dtype_t::int64) {
                auto mem = mkldnn::memory(
                  {{{dims},
                    dtype_to_mkldnn_memory_data_type(arr.dtype()),
                    format},
                   engine});
                std::transform(static_cast<int64_t*>(arr.data()),
                               static_cast<int64_t*>(arr.data()) +
                                 total_size(arr),
                               static_cast<int32_t*>(mem.get_data_handle()),
                               [](auto e) { return static_cast<int32_t>(e); });
                return mem;
            }
            return mkldnn::memory(
              {{{dims}, dtype_to_mkldnn_memory_data_type(arr.dtype()), format},
               engine},
              const_cast<void*>(arr.data()));
        }

        mkldnn::memory array_to_memory(array const& arr,
                                       mkldnn::memory::format format,
                                       mkldnn::engine const& engine) {
            return array_to_memory(arr, arr.dims(), format, engine);
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
