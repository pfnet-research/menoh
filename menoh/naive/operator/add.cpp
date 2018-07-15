#include <menoh/naive/operator/add.hpp>

#include <menoh/model_core.hpp>
#include <menoh/optional.hpp>
#include <menoh/utility.hpp>

#include <menoh/naive/operator/sum.hpp>

namespace menoh_impl {
    namespace naive_backend {

        computation_node_factory_return_type
        make_add(int32_t i, std::vector<node> const& node_list,
                 std::unordered_map<std::string, array> const& variable_table) {
            auto const& node = node_list.at(i);
            assert(node.input_name_list.size() == 2);

            return make_sum(i, node_list, variable_table);
        }

    } // namespace naive_backend
} // namespace menoh_impl
