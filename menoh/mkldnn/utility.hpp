#ifndef MENOH_MKLDNN_UTILITY_HPP
#define MENOH_MKLDNN_UTILITY_HPP

#include <vector>

// mkldnn.hpp requires including <string>
// c.f. https://stackoverflow.com/questions/19456626/compile-cln-with-clang-and-libc
#include <string>
#include <mkldnn.hpp>

#include <menoh/dtype.hpp>

namespace menoh_impl {
    class array;
} // namespace menoh_impl

namespace menoh_impl {
    namespace mkldnn_backend {

        std::vector<int> extract_dims(mkldnn::memory const& m);

        mkldnn::memory::data_type
        dtype_to_mkldnn_memory_data_type(dtype_t dtype);
        dtype_t mkldnn_memory_data_type_to_dtype(
          mkldnn::memory::data_type mem_data_type);

        mkldnn::memory array_to_memory(array const& arr,
                                       std::vector<int> const& dims,
                                       mkldnn::memory::format format,
                                       mkldnn::engine const& engine);

        mkldnn::memory array_to_memory(array const& arr,
                                       mkldnn::memory::format format,
                                       mkldnn::engine const& engine);

        array memory_to_array(mkldnn::memory const& mem);

    } // namespace mkldnn_backend
} // namespace menoh_impl

#endif // MENOH_MKLDNN_UTILITY_HPP
