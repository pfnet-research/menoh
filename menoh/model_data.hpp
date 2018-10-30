#ifndef MENOH_MODEL_DATA_HPP
#define MENOH_MODEL_DATA_HPP

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_set>
#include <vector>

#include <menoh/array.hpp>
#include <menoh/graph.hpp>

namespace menoh_impl {

    struct model_data {
        std::vector<node> node_list;
        std::vector<std::pair<std::string, array>>
          parameter_name_and_array_list;
    };

    inline model_data trim_redundant_nodes(
      menoh_impl::model_data model_data,
      std::vector<std::string> const& required_output_name_list) {

        auto needed_node_list = extract_needed_node_list(
          model_data.node_list, required_output_name_list);
        model_data.node_list =
          std::move(needed_node_list); // Update node_list in model_data here

        std::vector<std::string> all_parameter_name_list;
        all_parameter_name_list.reserve(
          model_data.parameter_name_and_array_list.size());
        std::transform(model_data.parameter_name_and_array_list.begin(),
                       model_data.parameter_name_and_array_list.end(),
                       std::back_inserter(all_parameter_name_list),
                       [](auto const& name_and_array_pair) {
                           return name_and_array_pair.first;
                       });

        auto all_input_name_set =
          extract_all_input_name_set(model_data.node_list);

        std::unordered_set<std::string> needed_parameter_name_set;
        std::sort(all_parameter_name_list.begin(),
                  all_parameter_name_list.end());
        std::set_intersection(
          all_parameter_name_list.begin(), all_parameter_name_list.end(),
          all_input_name_set.begin(), all_input_name_set.end(),
          std::inserter(needed_parameter_name_set,
                        needed_parameter_name_set.end()));
        auto new_end_iter = std::remove_if(
          model_data.parameter_name_and_array_list.begin(),
          model_data.parameter_name_and_array_list.end(),
          [&needed_parameter_name_set](
            auto const& parameter_name_and_array_pair) {
              auto const& parameter_name = parameter_name_and_array_pair.first;
              return needed_parameter_name_set.find(parameter_name) ==
                     needed_parameter_name_set.end();
          });
        model_data.parameter_name_and_array_list.erase(
          new_end_iter, model_data.parameter_name_and_array_list.end());
        return model_data;
    }

} // namespace menoh_impl

#endif // MENOH_MODEL_DATA_HPP
