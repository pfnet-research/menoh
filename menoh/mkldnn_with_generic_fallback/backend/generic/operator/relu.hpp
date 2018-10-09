#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_RELU_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_RELU_HPP

#include <menoh/mkldnn_with_generic_fallback/procedure.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace generic_backend {
            inline std::tuple<procedure,
                              std::vector<std::pair<std::string, array>>>
            make_relu(int node_index, std::vector<node> const& node_list,
                      std::vector<array> const& input_list,
                      std::unordered_map<std::string, array> const&
                        required_output_table) {
                assert(input_list.size() == 1);
                auto const& node = node_list.at(node_index);

                auto const& x_arr = input_list.at(0);

                auto found =
                  required_output_table.find(node.output_name_list.at(0));
                optional<array> output_opt;
                if(found == required_output_table.end()) {
                    output_opt = array(dtype_t::float_,
                                       x_arr.dims()); // TODO check inplace-able
                } else {
                    output_opt =
                      found->second; // output is required so not inplace-able
                }

                auto procedure = [x_arr, output = *output_opt]() {
                    for(decltype(total_size(x_arr)) i = 0;
                        i < total_size(x_arr); ++i) {
                        fat(output, i) = std::max(fat(x_arr, i), 0.f);
                    }
                };

                std::vector<std::pair<std::string, array>> outputs;
                if(found == required_output_table.end()) {
                    outputs.push_back(std::pair<std::string, array>(
                      node.output_name_list.at(0), *output_opt));
                }
                return std::make_tuple(procedure, outputs);
            }

        } // namespace generic_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_RELU_HPP
