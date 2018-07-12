#include <menoh/naive/operator/relu.hpp>

#include <numeric>

#include <menoh/model_core.hpp>
#include <menoh/optional.hpp>
#include <menoh/utility.hpp>

namespace menoh_impl {
    namespace naive_backend {

        computation_node_factory_return_type make_relu(
          int32_t i, std::vector<node> const& node_list,
          std::unordered_map<std::string, array> const& variable_table) {
            auto const& node = node_list.at(i);

            auto const& x_arr =
              find_value(variable_table, node.input_name_list.at(0));

            auto found = variable_table.find(node.output_name_list.at(0));
            optional<array> output_opt;
            if(found == variable_table.end()) {
                output_opt = array(dtype_t::float_,
                                   x_arr.dims()); // TODO check inplace-able
            } else {
                output_opt =
                  found->second; // output is required so not inplace-able
            }

            auto computation_node = [x_arr, output = *output_opt]() {
                for(decltype(total_size(x_arr)) i = 0; i < total_size(x_arr); ++i) {
                    fat(output, i) = std::max(fat(x_arr, i), 0.f);
                }
            };

            std::vector<std::pair<std::string, array>> outputs;
            if(found == variable_table.end()) {
                outputs.push_back(std::pair<std::string, array>(
                  node.output_name_list.at(0), *output_opt));
            }
            return std::make_tuple(computation_node, outputs);
        }

    } // namespace naive_backend
} // namespace menoh_impl
