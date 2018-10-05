#include <menoh/mkldnn_with_generic_fallback/backend/generic/generic_context.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/generic/operator.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace generic_backend {

            generic_context::generic_context() : context() {
                procedure_factory_table_.emplace("Relu", make_relu);
            }

            optional<std::tuple<std::vector<procedure>, int>>
            generic_context::do_process_node_list(
              std::string const& context_name, int current_index,
              std::vector<node> const& node_list,
              std::unordered_map<std::string, array> const&
                common_parameter_table,
              std::unordered_map<std::string, array> const& common_input_table,
              std::unordered_map<std::string, array> const&
                required_output_table,
              std::unordered_map<std::string, array_profile> const&
                output_profile_table,
              std::vector<
                std::pair<std::string, std::unique_ptr<context>>> const&
                context_list,
              logger_handle logger) {

                auto first_node_index = current_index;
                std::vector<procedure> procedure_list;

                std::vector<procedure> new_op_proc_list;

                for(; current_index < node_list.size(); ++current_index) {
                    auto const& node = node_list.at(current_index);
                    std::vector<array> input_list;
                    std::vector<procedure> new_copy_procedure_list;

                    for(auto const& input_name : node.input_name_list) {
                        do {
                            // search in self variable table
                            auto found_from_variable_table =
                              variable_table_.find(input_name);
                            if(found_from_variable_table !=
                               variable_table_.end()) {
                                *logger << input_name
                                        << " is found in self variable table"
                                        << std::endl;
                                input_list.push_back(
                                  found_from_variable_table->second);
                                break;
                            }

                            // search in common parameter and input table
                            auto found_from_common_table =
                              [&](auto const& table) {
                                  auto found = table.find(input_name);
                                  if(found != table.end()) {
                                      assert(found->second.dims().size() == 2 ||
                                             found->second.dims().size() == 4);
                                      input_list.push_back(found->second);
                                      return true;
                                  }
                                  return false;
                              };
                            if(found_from_common_table(
                                 common_parameter_table)) {
                                *logger << input_name
                                        << " is found in common parameter table"
                                        << std::endl;
                                break;
                            }
                            if(found_from_common_table(common_input_table)) {
                                *logger << input_name
                                        << " is found in common input table"
                                        << std::endl;
                                break;
                            }

                            // search in other contexts' variable table
                            bool is_found_from_other_context = false;
                            for(auto const& context_pair : context_list) {
                                if(context_pair.first == context_name) {
                                    continue; // skip self
                                }
                                auto found =
                                  context_pair.second->try_to_get_variable(
                                    input_name);
                                if(found) {
                                    *logger << input_name
                                            << " is found in other context's "
                                               "varibale table: "
                                            << context_pair.first << std::endl;
                                    procedure copy_proc;
                                    array arr;
                                    std::tie(copy_proc, arr) = *found;
                                    new_copy_procedure_list.push_back(
                                      copy_proc);
                                    input_list.push_back(arr);
                                    is_found_from_other_context = true;
                                    break;
                                }
                            }
                            assert(is_found_from_other_context);
                        } while(false);
                    }
                    std::vector<array> output_list;
                    for(auto const& output_name : node.output_name_list) {
                        auto found = required_output_table.find(output_name);
                        if(found == required_output_table.end()) {
                            // allocate new array by using profile
                            output_list.push_back(
                              array(output_profile_table.at(output_name)));
                        } else {
                            // use already allocated array
                            output_list.push_back(found->second);
                        }
                    }

                    procedure op_proc;
                    try {
                        auto factory =
                          procedure_factory_table_.at(node.op_type);
                        op_proc =
                          factory.operator()(node, input_list, output_list);
                    } catch(std::exception const& e) {
                        *logger << e.what() << std::endl;
                        break;
                    }
                    new_op_proc_list.push_back(op_proc);
                    procedure_list.insert(
                      procedure_list.end(),
                      std::make_move_iterator(new_copy_procedure_list.begin()),
                      std::make_move_iterator(new_copy_procedure_list.end()));

                    assert(node.output_name_list.size() == output_list.size());
                    for(int i = 0; i < node.output_name_list.size(); ++i) {
                        variable_table_.emplace(node.output_name_list.at(i),
                                                output_list.at(i));
                    }
                }

                // when no nodes are processed
                if(current_index == first_node_index) {
                    return nullopt;
                }

                procedure_list.insert(
                  procedure_list.end(),
                  std::make_move_iterator(new_op_proc_list.begin()),
                  std::make_move_iterator(new_op_proc_list.end()));

                return std::make_tuple(procedure_list, current_index);
            }
        } // namespace generic_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl
