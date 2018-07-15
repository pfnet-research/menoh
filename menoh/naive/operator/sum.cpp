#include <menoh/naive/operator/sum.hpp>

#include <algorithm>
#include <functional>

#include <menoh/model_core.hpp>
#include <menoh/optional.hpp>
#include <menoh/utility.hpp>

#include <menoh/naive/operator/index_conversion.hpp>

namespace menoh_impl {
    namespace naive_backend {

        computation_node_factory_return_type
        make_sum(int32_t i, std::vector<node> const& node_list,
                 std::unordered_map<std::string, array> const& variable_table) {
            auto const& node = node_list.at(i);

            std::vector<array> inputs;
            auto first_input =
              find_value(variable_table, node.input_name_list.front());
            inputs.push_back(first_input);
            for(int32_t i = 1; i < node.input_name_list.size(); ++i) {
                auto input =
                  find_value(variable_table, node.input_name_list.at(i));
                if(input.dims() != first_input.dims()) {
                    throw failed_to_configure_operator(
                      node.op_type, node.output_name_list.at(0),
                      "All inputs must have same dims: broadcast is not "
                      "supported yet");
                }
                inputs.push_back(input);
            }

            auto found = variable_table.find(node.output_name_list.front());
            optional<array> output_opt;
            if(found == variable_table.end()) {
                std::vector<int32_t> output_dims = inputs.front().dims();
                output_opt = zeros(dtype_t::float_, output_dims);
            } else {
                output_opt = found->second;
                std::fill(fbegin(*output_opt), fend(*output_opt), 0.);
            }

            auto computation_node = [inputs, output = *output_opt]() {
                for(auto const& input : inputs) {
                    std::transform(fbegin(output), fend(output), fbegin(input),
                                   fbegin(output), std::plus<>());
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
