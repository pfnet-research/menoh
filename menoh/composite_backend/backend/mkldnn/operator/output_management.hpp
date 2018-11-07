#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_OUTPUT_MANAGEMENT_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_OUTPUT_MANAGEMENT_HPP

#include <menoh/composite_backend/backend/mkldnn/formatted_array.hpp>
#include <menoh/composite_backend/backend/mkldnn/memory_cache.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace composite_backend {
        namespace mkldnn_backend {

            template <typename PrimitiveGen>
            inline std::tuple<memory_cache, std::vector<mkldnn::primitive>>
            manage_output(
              formatted_array const& output_formatted_array,
              mkldnn::memory::primitive_desc const& output_memory_pd,
              mkldnn::engine const& engine, PrimitiveGen pg) {
                std::vector<mkldnn::primitive> primitives;

                auto op_output_format = extract_format(output_memory_pd);
                bool is_reorder_needed =
                  !is_format_any(output_formatted_array) &&
                  output_formatted_array.format() != op_output_format;
                auto op_output_memory =
                  is_reorder_needed
                    ? mkldnn::memory(
                        output_memory_pd) // this line allocates memory
                    : output_formatted_array.make_memory(op_output_format,
                                                         engine);
                memory_cache output_memory_cache(op_output_memory);

                primitives.push_back(pg(op_output_memory));

                if(is_reorder_needed) {
                    auto output_memory =
                      make_memory(output_formatted_array, engine);
                    output_memory_cache.add_cached_memory(output_memory);
                    primitives.push_back(
                      mkldnn::reorder(op_output_memory, output_memory));
                }
                return std::make_tuple(output_memory_cache, primitives);
            }

            template <typename PrimitiveGen>
            inline memory_cache manage_output(
              formatted_array const& output_formatted_array,
              mkldnn::memory::primitive_desc const& output_memory_pd,
              mkldnn::engine const& engine,
              std::vector<mkldnn::primitive>& primitives, PrimitiveGen pg) {
                memory_cache output_memory_cache;
                std::vector<mkldnn::primitive> temp_primitives;
                std::tie(output_memory_cache, temp_primitives) = manage_output(
                  output_formatted_array, output_memory_pd, engine, pg);
                primitives.insert(primitives.end(), temp_primitives.begin(),
                                  temp_primitives.end());
                return output_memory_cache;
            }

        } // namespace mkldnn_backend
    }     // namespace composite_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_OUTPUT_MANAGEMENT_HPP
