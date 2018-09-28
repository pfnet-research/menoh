#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_RELU_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_RELU_HPP

#include <menoh/array.hpp>
#include <menoh/mkldnn_with_generic_fallback/procedure.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace generic_backend {
            inline procedure make_relu(node const& node,
                                       std::vector<array> const& input_list,
                                       std::vector<array> const& output_list) {
                assert(input_list.size() == 1);
                assert(output_list.size() == 1);

                auto procedure = [input = input_list.at(0),
                                  output = output_list.at(0)]() {
                    for(decltype(total_size(input)) i = 0;
                        i < total_size(input); ++i) {
                        fat(output, i) = std::max(fat(input, i), 0.f);
                    }
                };

                return procedure;
            }

        } // namespace generic_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_RELU_HPP
