#include <menoh/mkldnn/utility.hpp>

#include <cassert>

#include <menoh/array.hpp>
#include <menoh/dtype.hpp>
#include <menoh/exception.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        std::vector<int> extract_dims(mkldnn::memory const& m) {
            auto const& d = m.get_primitive_desc().desc().data;
            return std::vector<int>(d.dims, d.dims + d.ndims);
        }

        mkldnn::memory::data_type
        dtype_to_mkldnn_memory_data_type(dtype_t dtype) {
            // float16 and float64 is not supported by MKLDNN
            if(dtype == dtype_t::float32) {
                return mkldnn::memory::data_type::f32;
            }
            if(dtype == dtype_t::int8) {
                return mkldnn::memory::data_type::s8;
            }
            if(dtype == dtype_t::int16) {
                return mkldnn::memory::data_type::s16;
            }
            if(dtype == dtype_t::int32) {
                return mkldnn::memory::data_type::s32;
            }
            throw invalid_dtype(std::to_string(static_cast<int>(dtype)));
        }

        dtype_t mkldnn_memory_data_type_to_dtype(
          mkldnn::memory::data_type mem_data_type) {
            if(mem_data_type == mkldnn::memory::data_type::f32) {
                return dtype_t::float_;
            }
            if(mem_data_type == mkldnn::memory::data_type::s8) {
                return dtype_t::int8;
            }
            if(mem_data_type == mkldnn::memory::data_type::s16) {
                return dtype_t::int16;
            }
            if(mem_data_type == mkldnn::memory::data_type::s32) {
                return dtype_t::int32;
            }
            throw invalid_dtype(
              std::to_string(static_cast<int>(mem_data_type)));
        }

        mkldnn::memory array_to_memory(array const& arr,
                                       std::vector<int> const& dims,
                                       mkldnn::memory::format format,
                                       mkldnn::engine const& engine) {
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

        array memory_to_array(mkldnn::memory const& mem) {
            return array(mkldnn_memory_data_type_to_dtype(
                           static_cast<mkldnn::memory::data_type>(
                             mem.get_primitive_desc().desc().data.data_type)),
                         extract_dims(mem), mem.get_data_handle());
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
