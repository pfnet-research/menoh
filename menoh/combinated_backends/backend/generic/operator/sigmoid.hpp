#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_SIGMOID_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_SIGMOID_HPP

#include <cmath>

#include <menoh/array.hpp>
#include <menoh/combinated_backends/procedure.hpp>

namespace menoh_impl {
    namespace combinated_backends {
        namespace generic_backend {
            inline procedure
            make_sigmoid(node const&, std::vector<array> const& input_list,
                         std::vector<array> const& output_list) {
                assert(input_list.size() == 1);
                assert(output_list.size() == 1);

                auto input = input_list.at(0);
                if(input.dtype() != dtype_t::float_) {
                    throw std::runtime_error("invalid dtype");
                }

                auto procedure = [input, output = output_list.at(0)]() {
                    auto m = *std::max_element(fbegin(input), fend(input));
                    auto em = std::exp(-m);
                    for(decltype(total_size(input)) i = 0;
                        i < total_size(input); ++i) {
                        auto e = std::exp(fat(input, i) - m);
                        fat(output, i) = e / (e + em);
                    }
                };

                return procedure;
            }

        } // namespace generic_backend
    }     // namespace combinated_backends
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_BACKEND_GENERIC_OPERATOR_SIGMOID_HPP
