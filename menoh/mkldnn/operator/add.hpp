#ifndef MENOH_MKLDNN_OPERATOR_ADD_HPP
#define MENOH_MKLDNN_OPERATOR_ADD_HPP

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <mkldnn.hpp>

#include <menoh/array.hpp>
#include <menoh/node.hpp>

#include <menoh/mkldnn/primitive_factory_return_type.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        primitive_factory_return_type make_add_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, array> const& parameter_table,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine);

    } // namespace mkldnn_backend
} // namespace menoh_impl

#endif // MENOH_MKLDNN_OPERATOR_ADD_HPP
