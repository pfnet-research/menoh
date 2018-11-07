#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_MUL_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_MUL_HPP

#include <menoh/array.hpp>
#include <menoh/graph.hpp> // for dimension_mismatch error
#include <menoh/composite_backend/procedure.hpp>

namespace menoh_impl {
    namespace composite_backend {
        namespace generic_backend {
            inline procedure make_mul(node const& node,
                                      std::vector<array> const& input_list,
                                      std::vector<array> const& output_list) {
                assert(input_list.size() == 2);
                assert(output_list.size() == 1);

                for(auto const& input : input_list) {
                    if(input.dtype() != dtype_t::float_) {
                        throw invalid_dtype(
                          std::to_string(static_cast<int>(input.dtype())));
                    }
                }
                if(total_size(input_list.at(0)) !=
                   total_size(input_list.at(1))) {
                    throw dimension_mismatch(
                      node.op_type, node.output_name_list.front(),
                      "total size is invalid. broadcast is not supported yet",
                      std::to_string(total_size(input_list.at(0))),
                      std::to_string(total_size(input_list.at(1))));
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
    }     // namespace composite_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_MUL_HPP
