#include <menoh/naive/operator/gemm.hpp>

#include <numeric>

#include <menoh/model_core.hpp>
#include <menoh/optional.hpp>
#include <menoh/utility.hpp>

namespace menoh_impl {
    namespace naive_backend {

        computation_node_factory_return_type make_gemm(
          int32_t i, std::vector<node> const& node_list,
          std::unordered_map<std::string, array> const& variable_table) {
            auto const& node = node_list.at(i);

            auto alpha = optional_attribute_float(node, "alpha", 1.f);
            auto beta = optional_attribute_float(node, "beta", 1.f);
            auto trans_a = optional_attribute_int(node, "transA", 0);
            if(trans_a) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "transA of Gemm must be 0 but given: " +
                    std::to_string(alpha));
            }
            auto trans_b = optional_attribute_int(node, "transB", 0);
            if(!trans_b) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "transB of Gemm must be 1 but given: " +
                    std::to_string(alpha));
            }

            auto const& a_arr =
              find_value(variable_table, node.input_name_list.at(0));
            auto const& b_arr =
              find_value(variable_table, node.input_name_list.at(1));
            auto const& c_arr =
              find_value(variable_table, node.input_name_list.at(2));

            auto batch_size = a_arr.dims().at(0);
            auto input_size =
              std::accumulate(a_arr.dims().begin() + 1, a_arr.dims().end(), 1,
                              std::multiplies<>());
            auto output_size = b_arr.dims().at(0);

            auto found = variable_table.find(node.output_name_list.at(0));
            optional<array> output_opt;
            if(found == variable_table.end()) {
                output_opt = array(dtype_t::float_, {batch_size, output_size});
            } else {
                output_opt = found->second;
            }

            auto computation_node = [alpha, beta, batch_size, input_size,
                                     output_size, a_arr, b_arr, c_arr,
                                     output = *output_opt]() {
                for(int n = 0; n < batch_size; ++n) {
                    for(int o = 0; o < output_size; ++o) {
                        float sum = 0;
                        for(int i = 0; i < input_size; ++i) {
                            sum += alpha * fat(a_arr, n * input_size + i) *
                                   fat(b_arr, o * input_size + i);
                        }
                        sum += beta * fat(c_arr, o);
                        fat(output, n * output_size + o) = sum;
                    }
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
