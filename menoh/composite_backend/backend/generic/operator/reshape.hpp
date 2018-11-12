#ifndef MENOH_IMPL_COMPOSITE_BACKEND_BACKEND_GENERIC_OPERATOR_RESHAPE_HPP
#define MENOH_IMPL_COMPOSITE_BACKEND_BACKEND_GENERIC_OPERATOR_RESHAPE_HPP

#include <menoh/array.hpp>
#include <menoh/graph.hpp> // for dimension_mismatch error
#include <menoh/composite_backend/procedure.hpp>

namespace menoh_impl {
    namespace composite_backend {
        namespace generic_backend {
            inline procedure
            make_reshape(node const& node, std::vector<array> const& input_list,
                         std::vector<array> const& output_list) {
                assert(input_list.size() == 2);
                assert(output_list.size() == 1);

                auto procedure = [data = input_list.at(0),
                                  output = output_list.at(0)]() {
                    assert(total_size(data) == total_size(output));
                    std::copy(begin<dtype_t::float32>(data),
                              end<dtype_t::float32>(data),
                              begin<dtype_t::float32>(output));
                };

                return procedure;
            }

        } // namespace generic_backend
    }     // namespace composite_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_COMPOSITE_BACKEND_BACKEND_GENERIC_OPERATOR_RESHAPE_HPP
