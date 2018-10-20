#ifndef MENOH_IMPL_MKLDNN_WITH_FALLBACK_BACKEND_BACKEND_MKLDNN_MKLDNN_PROCEDURE_FACTORY_HPP
#define MENOH_IMPL_MKLDNN_WITH_FALLBACK_BACKEND_BACKEND_MKLDNN_MKLDNN_PROCEDURE_FACTORY_HPP

#include <string>
#include <utility>
#include <vector>

#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_cache.hpp>
#include <mkldnn.hpp>

#define MENOH_MKLDNN_CONTEXT_PROCEDURE_FACTORY(factory_name)           \
    factory_name(                                                      \
      node const& node,                                                \
      std::vector<std::reference_wrapper<memory_cache>> const&         \
        input_memory_cache_list,                                       \
      std::vector<formatted_array> const& output_formatted_array_list, \
      mkldnn::engine const& engine)

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            struct procedure_factory_return_type {
                std::vector<mkldnn::primitive> primitives;
                std::vector<memory_cache> output_memory_cache_list;
                std::vector<std::pair<std::string, memory_cache>>
                  named_temp_memory_cache_list;
            };

            using procedure_factory =
              std::function<procedure_factory_return_type(
                node const&,
                std::vector<std::reference_wrapper<
                  memory_cache>> const&, // input_memory_cache_list
                std::vector<
                  formatted_array> const&, // output_formatted_array_list
                mkldnn::engine const&      // engine
                )>;

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl
#endif // MENOH_IMPL_MKLDNN_WITH_FALLBACK_BACKEND_BACKEND_MKLDNN_MKLDNN_PROCEDURE_FACTORY_HPP
