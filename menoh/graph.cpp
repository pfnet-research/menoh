#include <menoh/graph.hpp>

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <numeric>
#include <unordered_map>

#include <menoh/exception.hpp>
#include <menoh/model_data.hpp>
#include <menoh/optional.hpp>
#include <menoh/utility.hpp>

#include <iostream>

namespace menoh_impl {

    std::vector<node> extract_needed_node_list(
      std::vector<node> const& node_list,
      std::vector<std::string> const& required_output_name_list) {
        std::unordered_set<std::string> required_output_name_set(
          required_output_name_list.begin(), required_output_name_list.end());
        std::vector<node> needed_node_list;
        while(!required_output_name_set.empty()) {
            std::unordered_set<std::string> next_required_output_name_set;
            for(auto const& required_output_name : required_output_name_set) {
                // Search node that issues required output
                auto needed_node_iter = std::find_if(
                  node_list.begin(), node_list.end(),
                  [&required_output_name](auto const& node) {
                      return std::any_of(
                        node.output_name_list.begin(),
                        node.output_name_list.end(),
                        [&required_output_name](auto const& output_name) {
                            return output_name == required_output_name;
                        });
                  });
                if (needed_node_iter==node_list.end())
                    continue;
                auto is_already_added =
                  std::find(needed_node_list.begin(), needed_node_list.end(),
                            *needed_node_iter) != needed_node_list.end();
                if(!is_already_added && needed_node_iter != node_list.end()) {
                    needed_node_list.push_back(*needed_node_iter);
                    next_required_output_name_set.insert(
                      needed_node_iter->input_name_list.begin(),
                      needed_node_iter->input_name_list.end());
                }
            }
            required_output_name_set = next_required_output_name_set;
        }
        return needed_node_list;
    }

    std::set<std::string>
    extract_all_input_name_set(std::vector<node> const& node_list) {
        std::set<std::string> all_input_name_set;
        for(auto const& node : node_list) {
            all_input_name_set.insert(node.input_name_list.begin(),
                                      node.input_name_list.end());
        }
        return all_input_name_set;
    }

    std::set<std::string>
    extract_all_output_name_set(std::vector<node> const& node_list) {
        std::set<std::string> all_output_name_set;
        for(auto const& node : node_list) {
            all_output_name_set.insert(node.output_name_list.begin(),
                                       node.output_name_list.end());
        }
        return all_output_name_set;
    }

    std::vector<std::string>
    name_set_difference(std::set<std::string> const& name_set_left,
                        std::set<std::string> const& name_set_right) {
        std::vector<std::string> diff_name_list;
        std::set_difference(name_set_left.begin(), name_set_left.end(),
                            name_set_right.begin(), name_set_right.end(),
                            std::back_inserter(diff_name_list));
        return diff_name_list;
    }

    auto extract_graph_input_name_list(std::vector<node> const& node_list) {
        auto all_input_name_set = extract_all_input_name_set(node_list);
        auto all_output_name_set = extract_all_output_name_set(node_list);
        return name_set_difference(all_input_name_set, all_output_name_set);
    }

    auto check_graph_computable(std::vector<node> const& node_list) {
        (void)node_list;
        return true; // TODO impl
    }

    graph::graph(std::vector<node>&& node_list)
      : node_list_(std::move(node_list)) {
        assert(check_graph_computable(node_list));
    }
    graph::graph(std::vector<node> const& node_list) : node_list_(node_list) {
        assert(check_graph_computable(node_list));
    }

    graph make_graph(std::vector<node> node_list) {
        auto all_output_name_set = extract_all_output_name_set(node_list);
        auto available_value_name_list =
          extract_graph_input_name_list(node_list);
        std::set<std::string> available_value_name_set(
          std::make_move_iterator(available_value_name_list.begin()),
          std::make_move_iterator(available_value_name_list.end()));
        std::vector<node> ordered_node_list;
        while(!node_list.empty()) {
            std::vector<node> next_node_list;
            auto next_available_value_name_set = available_value_name_set;
            std::vector<node> consumable_node_list;
            for(auto&& node : node_list) {
                std::set<std::string> unavailable_value_name_set;
                std::set<std::string> input_name_set(
                  node.input_name_list.begin(), node.input_name_list.end());
                std::set_difference(
                  input_name_set.begin(), input_name_set.end(),
                  available_value_name_set.begin(),
                  available_value_name_set.end(),
                  std::inserter(unavailable_value_name_set,
                                unavailable_value_name_set.end()));
                std::vector<std::string> lacking_input_name_list;
                std::set_intersection(
                  unavailable_value_name_set.begin(),
                  unavailable_value_name_set.end(), all_output_name_set.begin(),
                  all_output_name_set.end(),
                  std::back_inserter(lacking_input_name_list));
                if(lacking_input_name_list.empty()) {
                    next_available_value_name_set.insert(
                      node.output_name_list.begin(),
                      node.output_name_list.end());
                    consumable_node_list.push_back(std::move(node));
                } else {
                    next_node_list.push_back(std::move(node));
                }
            }
            node_list = std::move(next_node_list);
            available_value_name_set = std::move(next_available_value_name_set);
            ordered_node_list.insert(
              ordered_node_list.end(),
              std::make_move_iterator(consumable_node_list.begin()),
              std::make_move_iterator(consumable_node_list.end()));
        }
        return menoh_impl::graph(ordered_node_list);
    }

    auto
    extract_node_ref_that_has_specific_output(std::vector<node>& node_list,
                                              std::string const& output_name) {
        for(auto& node : node_list) {
            auto node_iter =
              std::find(node.output_name_list.begin(),
                        node.output_name_list.end(), output_name);
            if(node_iter != node.output_name_list.end()) {
                return optional<std::reference_wrapper<menoh_impl::node>>(
                  std::ref(node));
            }
        }
        return optional<std::reference_wrapper<node>>();
    }

    auto extract_node_ref_list_that_has_specific_input(
      std::vector<node>& node_list, std::string const& input_name) {
        std::vector<std::reference_wrapper<node>> node_ref_list;
        for(auto& node : node_list) {
            auto input_iter = std::find(node.input_name_list.begin(),
                                        node.input_name_list.end(), input_name);
            if(input_iter != node.input_name_list.end()) {
                node_ref_list.push_back(std::ref(node));
            }
        }
        return node_ref_list;
    }

    void trim_node(std::vector<node>& node_list, menoh_impl::node const& node) {
        assert(node.input_name_list.size() == 1 ||
               (node.op_type == "Reshape" && node.input_name_list.size() == 2));
        assert(
          node.output_name_list.size() == 1 ||
          (node.op_type == "Dropout" && node.output_name_list.size() == 2));
        auto next_node_ref_list = extract_node_ref_list_that_has_specific_input(
          node_list, node.output_name_list.at(0));
        for(menoh_impl::node& next_node : next_node_ref_list) {
            auto input_name_iter = std::find(next_node.input_name_list.begin(),
                                             next_node.input_name_list.end(),
                                             node.output_name_list.at(0));
            assert(input_name_iter != next_node.input_name_list.end());
            *input_name_iter = node.input_name_list.at(0);
        }
    }

    void trim_dropout(std::vector<node>& node_list) {
        reconstruct_node_list(
          node_list, [](std::vector<node>& node_list, auto const& node) {
              if(node.op_type == "Dropout") {
                  trim_node(node_list, node);
              }
          });
        node_list.erase(std::remove_if(node_list.begin(), node_list.end(),
                                       [](auto const& node) {
                                           return node.op_type == "Dropout";
                                       }),
                        node_list.end());
    }

    void trim_reshape(std::vector<node>& node_list) {
        reconstruct_node_list(
          node_list, [](std::vector<node>& node_list, auto const& node) {
              if(node.op_type == "Reshape") {
                  trim_node(node_list, node);
              }
          });
        node_list.erase(std::remove_if(node_list.begin(), node_list.end(),
                                       [](auto const& node) {
                                           return node.op_type == "Reshape";
                                       }),
                        node_list.end());
    }

    std::unordered_map<std::string, std::vector<int>> make_output_dims_table(
      menoh_impl::model_data const& model_data,
      std::vector<std::pair<std::string, std::vector<int>>> const&
        input_name_and_dims_pair_list) {

        std::vector<std::string> supported_operator_list{{"Abs",
                                                          "Elu",
                                                          "LeakyRelu",
                                                          "Relu",
                                                          "Sqrt",
                                                          "Tanh",
                                                          "AveragePool",
                                                          "Add",
                                                          "BatchNormalization",
                                                          "Concat",
                                                          "Conv",
                                                          "ConvTranspose",
                                                          "FC",
                                                          "Gemm",
                                                          "GlobalAveragePool",
                                                          "GlobalMaxPool",
                                                          "LRN",
                                                          "MaxPool",
                                                          "Softmax",
                                                          "Sum"}};

        std::unordered_map<std::string, std::vector<int>> variable_dims_table(
          input_name_and_dims_pair_list.begin(),
          input_name_and_dims_pair_list.end());
        auto graph = make_graph(model_data.node_list);
        auto parameter_table = std::unordered_map<std::string, array>(
          model_data.parameter_name_and_array_list.begin(),
          model_data.parameter_name_and_array_list.end());
        for(auto const& node : graph.node_list()) {
            if(node.op_type == "Conv") {
                auto weight_name = node.input_name_list.at(1);
                auto output_channel_num =
                  get_output_channel_num_from_parameter_dims(
                    find_value(parameter_table, weight_name).dims());
                auto output_dims = calc_2d_output_dims(node, output_channel_num,
                                                       variable_dims_table);
                auto dilations =
                  optional_attribute_ints(node, "dilations", {1, 1});
                if(dilations != std::vector<int>({1, 1})) {
                    auto actual = "(" + std::to_string(dilations.at(0)) + ", " +
                                  std::to_string(dilations.at(1)) + ")";
                    throw unsupported_operator_attribute(
                      node.op_type, node.output_name_list.front(), "dilations",
                      actual, "(1, 1)");
                }
                auto group = optional_attribute_int(node, "group", 1);
                if(group != 1) {
                    throw unsupported_operator_attribute(
                      node.op_type, node.output_name_list.front(), "group",
                      std::to_string(group), "1");
                }
                variable_dims_table.insert(
                  {node.output_name_list.at(0), output_dims});
            } else if(node.op_type == "MaxPool" ||
                      node.op_type == "AveragePool") {
                auto input_name = node.input_name_list.at(0);
                auto output_channel_num = get_channel_num_from_variable_dims(
                  find_value(variable_dims_table, input_name));
                if(node.op_type == "AveragePool") {
                    auto pads = optional_attribute_ints(node, "pads", {0, 0});
                    auto count_include_pad = optional_attribute_int(
                      node, "count_include_pad", 1); // TODO
                    if(pads != std::vector<int>({0, 0}) &&
                       count_include_pad == 0) {
                        throw unsupported_operator_attribute(
                          node.op_type, node.output_name_list.front(),
                          "count_include_pad",
                          std::to_string(count_include_pad), "0");
                    }
                }
                auto output_dims = calc_2d_output_dims(node, output_channel_num,
                                                       variable_dims_table);
                variable_dims_table.insert(
                  {node.output_name_list.at(0), output_dims});
            } else if(node.op_type == "GlobalMaxPool" ||
                      node.op_type == "GlobalAveragePool") {
                auto input_name = node.input_name_list.at(0);
                auto input_dims = find_value(variable_dims_table, input_name);
                auto output_dims = input_dims;
                output_dims.at(2) = 1;
                output_dims.at(3) = 1;
                variable_dims_table.insert(
                  {node.output_name_list.at(0), output_dims});
            } else if(node.op_type == "FC") {
                auto input_name = node.input_name_list.at(0);
                auto input_dims = find_value(variable_dims_table, input_name);
                auto batch_size = get_batch_size_from_variable_dims(
                  find_value(variable_dims_table, input_name));
                auto weight_dims =
                  find_value(parameter_table, node.input_name_list.at(1))
                    .dims();
                auto input_size =
                  std::accumulate(input_dims.begin() + 1, input_dims.end(), 1,
                                  std::multiplies<void>());
                if(input_size != weight_dims[1]) {
                    throw dimension_mismatch(
                      node.op_type, node.output_name_list.front(),
                      "input[1] and weight[1]", std::to_string(input_size),
                      std::to_string(weight_dims[1]));
                }
                std::vector<int> output_dims{batch_size, weight_dims[0]};
                variable_dims_table.insert(
                  {node.output_name_list.at(0), output_dims});
            } else if(node.op_type == "Gemm") {
                auto input_name = node.input_name_list.at(0);
                auto input_dims = find_value(variable_dims_table, input_name);
                auto batch_size = get_batch_size_from_variable_dims(
                  find_value(variable_dims_table, input_name));
                auto weight_dims =
                  find_value(parameter_table, node.input_name_list.at(1))
                    .dims();
                auto trans_a = optional_attribute_int(node, "transA", 0);
                if(trans_a) {
                    throw unsupported_operator_attribute(
                      node.op_type, node.output_name_list.front(), "transA",
                      std::to_string(trans_a), "0");
                }
                auto trans_b = optional_attribute_int(node, "transB", 0);
                if(!trans_b) {
                    throw unsupported_operator_attribute(
                      node.op_type, node.output_name_list.front(), "transB",
                      std::to_string(trans_b), "1");
                }
                auto input_size =
                  std::accumulate(input_dims.begin() + 1, input_dims.end(), 1,
                                  std::multiplies<void>());
                if(input_size != weight_dims[1]) {
                    throw dimension_mismatch(
                      node.op_type, node.output_name_list.front(),
                      "input[1] and weight[1]", std::to_string(input_size),
                      std::to_string(weight_dims[1]));
                }
                std::vector<int> output_dims{batch_size, weight_dims[0]};
                variable_dims_table.insert(
                  {node.output_name_list.at(0), output_dims});
            } else if(std::find(supported_operator_list.begin(),
                                supported_operator_list.end(), node.op_type) !=
                      supported_operator_list
                        .end()) { // check if supported operator
                auto input_name = node.input_name_list.at(0);
                auto output_dims = find_value(variable_dims_table, input_name);
                variable_dims_table.insert(
                  {node.output_name_list.at(0), output_dims});
            }
            else {
                throw unsupported_operator(node.op_type);
            }
        }
        return variable_dims_table;
    } // namespace menoh_impl

} // namespace menoh_impl
