#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_SOFTMAX_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_SOFTMAX_HPP

#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_cache.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            inline std::tuple<std::vector<mkldnn::primitive>,
                              std::vector<std::pair<std::string, memory_cache>>>
            make_softmax(
              node const& node,
              std::vector<std::reference_wrapper<memory_cache>> const&
                input_memory_cache_list,
              std::vector<mkldnn::memory> const& output_memory_list,
              mkldnn::engine const& engine) {

                std::vector<mkldnn::primitive> primitives;

                auto axis = attribute_int(node, "axis");

                memory_cache& input_memory_cache =
                  input_memory_cache_list.at(0);
                auto input_dims = input_memory_cache.dims();
                auto input_memory = input_memory_cache.get_data_memory();
                auto input_format = extract_format(input_memory);

                auto const& output_memory = output_memory_list.at(0);
                auto output_dims = extract_dims(output_memory);
                assert(output_dims.at(0) == input_dims.at(0) &&
                       "invalid shape inference");

                mkldnn::softmax_forward::desc softmax_desc(
                  mkldnn::prop_kind::forward_inference,
                  input_memory.get_primitive_desc().desc(), axis);
                auto softmax_pd =
                  mkldnn::softmax_forward::primitive_desc(softmax_desc, engine);

                memory_cache output_memory_cache;
                optional<mkldnn::memory> op_output_memory;
                if(extract_format(output_memory) ==
                     mkldnn::memory::format::any ||
                   extract_format(output_memory) == input_format) {
                    op_output_memory =
                      mkldnn::memory({{{extract_dims(output_memory)},
                                       extract_data_type(output_memory),
                                       input_format},
                                      engine},
                                     output_memory.get_data_handle());
                    output_memory_cache.add_cached_memory(*op_output_memory);
                } else {
                    op_output_memory =
                      mkldnn::memory({{{extract_dims(output_memory)},
                                       extract_data_type(output_memory),
                                       input_format},
                                      engine});
                    output_memory_cache.add_cached_memory(*op_output_memory);
                    output_memory_cache.add_cached_memory(output_memory);
                }

                primitives.push_back(mkldnn::softmax_forward(
                  softmax_pd, input_memory, *op_output_memory));

                if(extract_format(output_memory) !=
                     mkldnn::memory::format::any &&
                   extract_format(output_memory) != input_format) {
                    primitives.push_back(
                      mkldnn::reorder(*op_output_memory, output_memory));
                }

                std::vector<std::pair<std::string, memory_cache>>
                  output_memory_cache_list({std::make_pair(
                    node.output_name_list.at(0), output_memory_cache)});
                return std::make_tuple(primitives, output_memory_cache_list);
            }

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_MKLDNN_OPERATOR_SOFTMAX_HPP
