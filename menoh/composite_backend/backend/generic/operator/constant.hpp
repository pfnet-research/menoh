#ifndef MENOH_IMPL_COMPOSITE_BACKEND_BACKEND_GENERIC_OPERATOR_CONSTANT_HPP
#define MENOH_IMPL_COMPOSITE_BACKEND_BACKEND_GENERIC_OPERATOR_CONSTANT_HPP

#include <menoh/array.hpp>
#include <menoh/composite_backend/procedure.hpp>

#include <iostream>

namespace menoh_impl {
    namespace composite_backend {
        namespace generic_backend {
            inline procedure
            make_constant(node const& node,
                          std::vector<array> const& input_list,
                          std::vector<array> const& output_list) {
                assert(input_list.size() == 0);
                assert(output_list.size() == 1);
                array value = attribute_tensor(node, "value");
                std::cout << total_size(value) << std::endl;
                for(decltype(total_size(value)) i = 0; i < total_size(value);
                    ++i) {
                    std::cout << fat(value, i) << std::endl;
                    fat(output_list[0], i) = fat(value, i);
                }
                auto procedure = []() {};

                return procedure;
            }

        } // namespace generic_backend
    }     // namespace composite_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_COMPOSITE_BACKEND_BACKEND_GENERIC_OPERATOR_CONSTANT_HPP
