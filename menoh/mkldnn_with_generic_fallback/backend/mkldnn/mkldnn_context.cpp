#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/mkldnn_context.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/operator.hpp>

#include <menoh/mkldnn/utility.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            mkldnn_context::mkldnn_context() : context() {
                procedure_factory_table_.emplace(
                  "Gemm", mkldnn_with_generic_fallback_backend::mkldnn_backend::
                            make_gemm);
            }

            optional<std::tuple<std::vector<procedure>, int>>
            mkldnn_context::do_process_node_list(
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

                std::vector<mkldnn::primitive> primitive_list;

                for(; current_index < node_list.size(); ++current_index) {
                    auto const& node = node_list.at(current_index);
                    std::vector<array> input_list;
                    std::vector<procedure> new_copy_procedure_list;

                    for(auto const& input_name : node.input_name_list) {
                        do {
                            // search in self variable table
                            auto found_from_variable_memory_table =
                              variable_memory_table_.find(input_name);
                            if(found_from_variable_memory_table !=
                               variable_memory_table_.end()) {
                                *logger << input_name
                                        << " is found in self variable table"
                                        << std::endl;
                                input_list.push_back(
                                  menoh_impl::mkldnn_backend::memory_to_array(
                                    found_from_variable_memory_table->second));
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
                                    procedure copy_proc; // copy from other
                                                         // context to array
                                    array arr;
                                    std::tie(copy_proc, arr) = *found;
                                    assert(arr.dims().size() == 2 ||
                                           arr.dims().size() == 4);
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

                    // make primitives and
                    std::vector<mkldnn::primitive> new_primitive_list;
                    std::vector<std::pair<std::string, mkldnn::memory>>
                      new_output_memory_list;
                    std::vector<mkldnn::memory> new_temp_memory_list;
                    try {
                        auto factory =
                          procedure_factory_table_.at(node.op_type);
                        std::tie(new_primitive_list, new_output_memory_list,
                                 new_temp_memory_list) =
                          factory.operator()(current_index, node_list,
                                             input_list, required_output_table,
                                             engine_);
                    } catch(...) { break; }
                    primitive_list.insert(primitive_list.end(),
                                          new_primitive_list.begin(),
                                          new_primitive_list.end());
                    // add copy procedures
                    procedure_list.insert(
                      procedure_list.end(),
                      std::make_move_iterator(new_copy_procedure_list.begin()),
                      std::make_move_iterator(new_copy_procedure_list.end()));

                    // update context
                    variable_memory_table_.insert(
                      std::make_move_iterator(new_output_memory_list.begin()),
                      std::make_move_iterator(new_output_memory_list.end()));
                    temp_memory_list_.insert(
                      temp_memory_list_.end(),
                      std::make_move_iterator(new_temp_memory_list.begin()),
                      std::make_move_iterator(new_temp_memory_list.end()));
                }

                // when no nodes are processed
                if(current_index == first_node_index) {
                    return nullopt;
                }

                procedure_list.emplace_back([this, primitive_list]() {
                    mkldnn::stream(mkldnn::stream::kind::eager)
                      .submit(primitive_list)
                      .wait();
                });

                return std::make_tuple(procedure_list, current_index);
            }

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl
