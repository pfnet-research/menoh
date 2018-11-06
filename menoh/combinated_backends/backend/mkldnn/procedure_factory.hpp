#ifndef MENOH_IMPL_COMBINATED_BACKENDS_BACKEND_MKLDNN_MKLDNN_PROCEDURE_FACTORY_HPP
#define MENOH_IMPL_COMBINATED_BACKENDS_BACKEND_MKLDNN_MKLDNN_PROCEDURE_FACTORY_HPP

#include <string>
#include <utility>
#include <vector>

#include <menoh/combinated_backends/backend/mkldnn/memory_cache.hpp>
#include <mkldnn.hpp>

#define MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_PARAMETER_LIST          \
    node const &node,                                                  \
      std::vector<std::reference_wrapper<memory_cache>> const          \
        &input_memory_cache_list,                                      \
      std::vector<formatted_array> const &output_formatted_array_list, \
      mkldnn::engine const &engine

#define MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY_ARGUMENT_LIST \
    node, input_memory_cache_list, output_formatted_array_list, engine

namespace menoh_impl {
    namespace combinated_backends {
        namespace mkldnn_backend {

            struct procedure_factory_return_type {
                std::vector<mkldnn::primitive> primitives;
                std::vector<memory_cache> output_memory_cache_list;
                std::vector<std::pair<std::string, memory_cache>>
                  named_temp_memory_cache_list;
            };

            using procedure_factory =
              std::function<procedure_factory_return_type(
                node const &,
                std::vector<std::reference_wrapper<memory_cache>> const
                  &, // input_memory_cache_list
                std::vector<formatted_array> const
                  &,                   // output_formatted_array_list
                mkldnn::engine const & // engine
                )>;

        } // namespace mkldnn_backend
    }     // namespace combinated_backends
} // namespace menoh_impl
#endif // MENOH_IMPL_COMBINATED_BACKENDS_BACKEND_MKLDNN_MKLDNN_PROCEDURE_FACTORY_HPP
