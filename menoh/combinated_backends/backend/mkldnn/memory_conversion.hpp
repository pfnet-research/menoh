#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_MKLDNN_MEMORY_CONVERSION_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_MKLDNN_MEMORY_CONVERSION_HPP

#include <menoh/array.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace combinated_backends {
        namespace mkldnn_backend {

            bool is_data_format(mkldnn::memory::format format);

            std::vector<int> extract_dims(mkldnn::memory const& m);
            mkldnn::memory::format extract_format(mkldnn::memory const& m);
            mkldnn::memory::data_type
            extract_data_type(mkldnn::memory const& m);

            mkldnn::memory::data_type
            dtype_to_mkldnn_memory_data_type(dtype_t dtype);

            dtype_t mkldnn_memory_data_type_to_dtype(
              mkldnn::memory::data_type mem_data_type);

            mkldnn::memory::format ndims_to_data_memory_format(int ndims);
            mkldnn::memory::format ndims_to_weight_memory_format(int ndims);

            mkldnn::memory array_to_memory(array const& arr,
                                           std::vector<int> const& dims,
                                           mkldnn::memory::format format,
                                           mkldnn::engine const& engine);

            mkldnn::memory array_to_memory(array const& arr,
                                           mkldnn::memory::format format,
                                           mkldnn::engine const& engine);

            mkldnn::memory array_to_data_memory(array const& arr,
                                                mkldnn::engine const& engine);

            array memory_to_array(mkldnn::memory const& mem);

            mkldnn::memory
            make_memory_from_array_profile(array_profile const& profile,
                                           mkldnn::memory::format format,
                                           mkldnn::engine const& engine);

            mkldnn::memory
            make_data_memory_from_array_profile(array_profile const& profile,
                                                mkldnn::engine const& engine);

        } // namespace mkldnn_backend
    }     // namespace combinated_backends
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_MKLDNN_MEMORY_CONVERSION_HPP
