#ifndef MENOH_MKLDNN_PRIMITIVE_FACTORY_RETURN_TYPE_HPP
#define MENOH_MKLDNN_PRIMITIVE_FACTORY_RETURN_TYPE_HPP

#include <tuple>
#include <unordered_map>
#include <vector>

#include <mkldnn.hpp>

#include <menoh/array.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        using primitive_factory_return_type =
          std::tuple<std::vector<mkldnn::primitive>, // net
                     std::unordered_map<std::string,
                                        mkldnn::memory>, // output_memory_table
                     std::vector<mkldnn::memory>,        // temp_memory_list
                     std::vector<array>                  // owned_array_list
                     >;
    } // namespace mkldnn_backend
} // namespace menoh_impl

#endif // MENOH_MKLDNN_PRIMITIVE_FACTORY_RETURN_TYPE_HPP
