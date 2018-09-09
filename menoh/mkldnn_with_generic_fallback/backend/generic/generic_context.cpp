#include <menoh/mkldnn_with_generic_fallback/backend/generic/generic_context.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/generic/operator.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace generic_backend {

            generic_context::generic_context() : context() {
                procedure_factory_table_.emplace("Relu", make_relu);
            }

            optional<std::function<void()>>
            generic_context::try_to_get_input_from_common_table(
              std::string const& input_name,
              std::unordered_map<std::string, array> const& common_table) {
                auto found = common_table.find(input_name);
                if(found != common_table.end()) {
                    variable_table_.emplace(input_name, found->second);
                    return std::function<void()>([this, &input_name]() {
                        variable_table_.erase(input_name);
                    });
                }
                return nullopt;
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
              std::vector<
                std::pair<std::string, std::unique_ptr<context>>> const&
                context_list,
              logger_handle logger) {
                auto const& node = node_list.at(current_index);
                std::vector<procedure> procedure_list;
                std::vector<array> input_list;
                std::vector<std::function<void()>> rollback_list;
                for(auto const& input_name : node.input_name_list) {
                    if(variable_table_.find(input_name) !=
                       variable_table_.end()) {
                        *logger << input_name
                                << " is found in self variable table"
                                << std::endl;
                        continue; // when found
                    }
                    do {
                        {
                            // normally, copy once array here, because
                            // parameter is statically fixed
                            auto rollback = try_to_get_input_from_common_table(
                              input_name, common_parameter_table);
                            if(rollback) {
                                *logger << input_name
                                        << " is found in common parameter table"
                                        << std::endl;
                                rollback_list.emplace_back(*rollback);
                                break;
                            }
                        }
                        {
                            // normally, allocate buffer for variable and
                            // issue copy procedure
                            auto rollback = try_to_get_input_from_common_table(
                              input_name, common_input_table);
                            if(rollback) {
                                *logger << input_name
                                        << " is found in common input table"
                                        << std::endl;
                                rollback_list.emplace_back(*rollback);
                                break;
                            }
                        }

                        // take from other context
                        bool is_found_from_other_context = false;
                        for(auto const& context_pair : context_list) {
                            std::string name = context_pair.first;
                            context* con = context_pair.second.get();
                            if(name == context_name) {
                                continue; // skip self
                            }
                            auto found = con->try_to_get_variable(input_name);
                            if(found) {
                                *logger << input_name
                                        << " is found in other context's "
                                           "varibale table: "
                                        << context_pair.first << std::endl;
                                procedure proc;
                                array arr;
                                std::tie(proc, arr) = *found;
                                procedure_list.push_back(proc);
                                variable_table_.emplace(input_name, arr);
                                rollback_list.push_back([this, &input_name]() {
                                    variable_table_.erase(input_name);
                                });
                                is_found_from_other_context = true;
                                break;
                            }
                        }
                        assert(is_found_from_other_context);
                    } while(false);
                    assert(variable_table_.find(input_name) !=
                           variable_table_.end());
                    input_list.push_back(variable_table_.at(input_name));
                }
                procedure proc;
                std::vector<std::pair<std::string, array>> new_variables;
                try {
                    auto factory = procedure_factory_table_.at(node.op_type);
                    std::tie(proc, new_variables) =
                      factory.operator()(current_index, node_list, input_list,
                                         required_output_table);
                } catch(...) {
                    for(auto const& rollback : rollback_list) {
                        rollback(); // remove new inputs
                    }
                    return nullopt;
                }
                procedure_list.push_back(std::move(proc));
                variable_table_.insert(
                  std::make_move_iterator(new_variables.begin()),
                  std::make_move_iterator(new_variables.end()));
                return std::make_tuple(procedure_list, current_index + 1);
            }
        } // namespace generic_backend

    } // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl
