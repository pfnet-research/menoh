#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_TRANSPOSE_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_TRANSPOSE_HPP

#include <vector>
#include <menoh/array.hpp>
#include <menoh/mkldnn_with_generic_fallback/procedure.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace generic_backend {

            inline std::vector<int>
            calc_strides(std::vector<int> const& dims) {
                std::vector<int> strides({1});
                for(int i = dims.size() - 1; i >= 1; --i) {
                    strides.push_back(strides.back() * dims.at(i));
                }
                std::reverse(strides.begin(), strides.end());
                return strides;
            }

            inline std::vector<int>
            index_to_indices(int index, std::vector<int> const& strides) {
                std::vector<int> indices;
                for(auto s : strides) {
                    indices.push_back(index / s);
                    index %= s;
                }
                return indices;
            }

            inline int indices_to_index(std::vector<int> const& indices,
                                        std::vector<int> const& strides) {
                int index = 0;
                for(int i = 0; i < indices.size(); ++i) {
                    index += indices.at(i) * strides.at(i);
                }
                return index;
            }

            inline procedure
            make_transpose(node const& node, std::vector<array> const& input_list,
                           std::vector<array> const& output_list) {
                assert(input_list.size() == 1);
                assert(output_list.size() == 1);

                auto input = input_list.at(0);
                if(input.dtype() != dtype_t::float_) {
                    throw std::runtime_error("not implemented yet");
                }

                auto perm = attribute_ints(node, "perm");
                auto output = output_list.at(0);
                auto input_strides = calc_strides(input.dims());
                auto output_strides = calc_strides(output.dims());

                auto procedure = [input, output, input_strides, output_strides,
                                  perm]() {
                    for(decltype(total_size(input)) i = 0;
                        i < total_size(input); ++i) {
                        auto input_indices = index_to_indices(i, input_strides);
                        std::vector<int> output_indices;
                        for(auto p : perm) {
                            output_indices.push_back(input_indices.at(p));
                        }
                        fat(output,
                            indices_to_index(output_indices, output_strides)) =
                          fat(input,
                              indices_to_index(input_indices, input_strides));
                    }
                };

                return procedure;
            }

        } // namespace generic_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_TRANSPOSE_HPP
