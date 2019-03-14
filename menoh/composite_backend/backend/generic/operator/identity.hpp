#ifndef MENOH_IMPL_COMPOSITE_BACKEND_BACKEND_GENERIC_OPERATOR_IDENTITY_HPP
#define MENOH_IMPL_COMPOSITE_BACKEND_BACKEND_GENERIC_OPERATOR_IDENTITY_HPP

#include <algorithm>

#include <menoh/array.hpp>
#include <menoh/composite_backend/procedure.hpp>

namespace menoh_impl {
    namespace composite_backend {
        namespace generic_backend {
            inline procedure
            make_identity(node const& node,
                          std::vector<array> const& input_list,
                          std::vector<array> const& output_list) {
                assert(input_list.size() == 1);
                assert(output_list.size() == 1);

                auto procedure = [input = input_list.at(0),
                                  output = output_list.at(0)]() {
                    for(decltype(total_size(input)) i = 0;
                        i < total_size(input); ++i) {
                        fat(output, i) = fat(input, i);
                        std::copy(
                          static_cast<char*>(input.data()),
                          static_cast<char*>(
                            input.data() + total_size(input) *
                                             get_size_in_bytes(input.dtype())),
                          static_cast<char*>(output.data()));
                    }
                };

                return procedure;
            }

        } // namespace generic_backend
    }     // namespace composite_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_COMPOSITE_BACKEND_BACKEND_GENERIC_OPERATOR_IDENTITY_HPP
