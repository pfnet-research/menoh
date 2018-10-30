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
                if(needed_node_iter == node_list.end()) {
                    continue;
                }
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

} // namespace menoh_impl
