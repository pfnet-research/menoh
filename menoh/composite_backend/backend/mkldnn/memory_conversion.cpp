#include <menoh/composite_backend/backend/mkldnn/memory_conversion.hpp>

namespace menoh_impl {
    namespace composite_backend {
        namespace mkldnn_backend {

            bool is_data_format(mkldnn::memory::format format) {
                std::vector<mkldnn::memory::format> data_memory_formats(
                  {mkldnn::memory::format::x, //
                   mkldnn::memory::format::nc,
                   // mkldnn::memory::format::ncw,
                   // mkldnn::memory::format::nwc,
                   // mkldnn::memory::format::nCw16c,
                   mkldnn::memory::format::nchw,
                   // mkldnn::memory::format::nhwc,
                   mkldnn::memory::format::chwn,
                   // mkldnn::memory::format::nCw8c,
                   mkldnn::memory::format::nChw8c,
                   mkldnn::memory::format::nChw16c,
                   mkldnn::memory::format::ncdhw, //
                   mkldnn::memory::format::ndhwc,
                   // mkldnn::memory::format::nCdhw8c,
                   mkldnn::memory::format::nCdhw16c});
                return std::find(data_memory_formats.begin(),
                                 data_memory_formats.end(),
                                 format) != data_memory_formats.end();
            }

            std::vector<int> extract_dims(mkldnn::memory const& m) {
                auto const& d = m.get_primitive_desc().desc().data;
                return std::vector<int>(d.dims, d.dims + d.ndims);
            }

            mkldnn::memory::format extract_format(mkldnn::memory const& m) {
                auto const& d = m.get_primitive_desc().desc().data;
                return static_cast<mkldnn::memory::format>(d.format);
            }

            mkldnn::memory::data_type
            extract_data_type(mkldnn::memory const& m) {
                auto const& d = m.get_primitive_desc().desc().data;
                return static_cast<mkldnn::memory::data_type>(d.data_type);
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

            mkldnn::memory::format ndims_to_data_memory_format(int ndims) {
                if(ndims == 1) {
                    return mkldnn::memory::format::x;
                }
                if(ndims == 2) {
                    return mkldnn::memory::format::nc;
                }
                /*
                if(ndims == 3) {
                    return mkldnn::memory::format::ncw;
                }
                */
                if(ndims == 4) {
                    return mkldnn::memory::format::nchw;
                }
                throw std::runtime_error("ndims_to_data_memory_format: invalid ndims: " + std::to_string(ndims));
            }

            mkldnn::memory::format ndims_to_weight_memory_format(int ndims) {
                if(ndims == 1) {
                    return mkldnn::memory::format::x;
                }
                if(ndims == 2) {
                    return mkldnn::memory::format::oi;
                }
                /*
                if(ndims == 3) {
                    return mkldnn::memory::format::oiw;
                }
                */
                if(ndims == 4) {
                    return mkldnn::memory::format::oihw;
                }
                throw std::runtime_error("ndims_to_weight_memory_format: invalid ndims" + std::to_string(ndims));
            }

            mkldnn::memory array_to_memory(array const& arr,
                                           std::vector<int> const& dims,
                                           mkldnn::memory::format format,
                                           mkldnn::engine const& engine) {
                return mkldnn::memory(
                  {{{dims},
                    dtype_to_mkldnn_memory_data_type(arr.dtype()),
                    format},
                   engine},
                  const_cast<void*>(arr.data()));
            }

            mkldnn::memory array_to_memory(array const& arr,
                                           mkldnn::memory::format format,
                                           mkldnn::engine const& engine) {
                return array_to_memory(arr, arr.dims(), format, engine);
            }

            mkldnn::memory array_to_data_memory(array const& arr,
                                                mkldnn::engine const& engine) {
                return array_to_memory(
                  arr, arr.dims(),
                  ndims_to_data_memory_format(arr.dims().size()), engine);
            }

            array memory_to_array(mkldnn::memory const& mem) {
                assert(extract_format(mem) == mkldnn::memory::format::x ||
                       extract_format(mem) == mkldnn::memory::format::nc ||
                       // extract_format(mem) == mkldnn::memory::format::ncw ||
                       extract_format(mem) == mkldnn::memory::format::nchw);
                return array(
                  mkldnn_memory_data_type_to_dtype(
                    static_cast<mkldnn::memory::data_type>(
                      mem.get_primitive_desc().desc().data.data_type)),
                  extract_dims(mem), mem.get_data_handle());
            }

            mkldnn::memory
            make_memory_from_array_profile(array_profile const& profile,
                                           mkldnn::memory::format format,
                                           mkldnn::engine const& engine) {
                return mkldnn::memory(
                  {{{profile.dims()},
                    dtype_to_mkldnn_memory_data_type(profile.dtype()),
                    format},
                   engine});
            }

            mkldnn::memory
            make_data_memory_from_array_profile(array_profile const& profile,
                                                mkldnn::engine const& engine) {
                return make_memory_from_array_profile(
                  profile, ndims_to_data_memory_format(profile.dims().size()),
                  engine);
            }

        } // namespace mkldnn_backend
    }     // namespace composite_backend
} // namespace menoh_impl
