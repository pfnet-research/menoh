#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_MUL_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_MUL_HPP

#include <menoh/array.hpp>
#include <menoh/mkldnn_with_generic_fallback/procedure.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace generic_backend {
            inline procedure make_mul(node const&,
                                      std::vector<array> const& input_list,
                                      std::vector<array> const& output_list) {
                assert(input_list.size() == 2);
                assert(output_list.size() == 1);

                for(auto const& input : input_list) {
                    if(input.dtype() != dtype_t::float_) {
                        throw std::runtime_error("invalid dtype");
                    }
                }
                if(total_size(input_list.at(0)) != total_size(input_list.at(1))) {
                    throw std::runtime_error("broadcast is not supported yet");
                }

                auto procedure = [input_a = input_list.at(0),
                                  input_b = input_list.at(1),
                                  output = output_list.at(0)]() {
                    for(decltype(total_size(input_a)) i = 0;
                        i < total_size(input_a); ++i) {
                        fat(output, i) = fat(input_a, i) * fat(input_b, i);
                    }
                };

                return procedure;
            }

        } // namespace generic_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_MUL_HPP
