#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_conversion.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/mkldnn_context.hpp>
#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/operator.hpp>

#include <menoh/graph.hpp> // for unsupported_operator error

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            mkldnn_context::mkldnn_context() : context() {
                using namespace mkldnn_with_generic_fallback_backend::
                  mkldnn_backend;
              
                // BatchNormalization
                procedure_factory_table_.emplace(
                  "BatchNormalization", mkldnn_backend::make_batch_norm);

                // Conv
                procedure_factory_table_.emplace("Conv", make_conv);
              
                // Gemm
                procedure_factory_table_.emplace("Gemm", make_gemm);

                // Eltwise
                procedure_factory_table_.emplace("Abs", make_abs);
                procedure_factory_table_.emplace("Elu", make_elu);
                procedure_factory_table_.emplace("LeakyRelu", make_leaky_relu);
                procedure_factory_table_.emplace("Relu", make_relu);
                procedure_factory_table_.emplace("Sqrt", make_sqrt);
                procedure_factory_table_.emplace("Tanh", make_tanh);
              
                // Pool
                procedure_factory_table_.emplace("AveragePool", make_average_pool);
                procedure_factory_table_.emplace("MaxPool", make_max_pool);
              
                // Softmax
                procedure_factory_table_.emplace("Softmax", make_softmax);

                // Sum and Add
                procedure_factory_table_.emplace("Sum", make_sum);
                procedure_factory_table_.emplace("Add", make_sum);
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
                static_cast<void>(output_profile_table); // maybe unused

                auto first_node_index = current_index;
                std::vector<procedure> procedure_list;

                std::vector<mkldnn::primitive> primitive_list;

                for(; current_index < static_cast<int>(node_list.size());
                    ++current_index) {
                    auto const& node = node_list.at(current_index);

                    std::vector<procedure> new_copy_procedure_list;
                    std::vector<mkldnn::primitive> new_primitive_list;
                    std::vector<memory_cache> new_output_memory_cache_list;
                    std::vector<std::pair<std::string, memory_cache>>
                      new_named_temp_memory_cache_list;

                    try {
                        std::vector<std::reference_wrapper<memory_cache>>
                          input_memory_cache_list;
                        for(auto const& input_name : node.input_name_list) {
                            do {
                                // search in self variable table
                                {
                                    auto found =
                                      variable_memory_cache_table_.find(
                                        input_name);
                                    if(found !=
                                       variable_memory_cache_table_.end()) {
                                        *logger << input_name
                                                << " is found from self "
                                                   "variable table"
                                                << std::endl;
                                        input_memory_cache_list.push_back(
                                          found->second);
                                        break;
                                    }
                                }

                                // search in common parameter table
                                {
                                    auto found =
                                      common_parameter_table.find(input_name);
                                    if(found != common_parameter_table.end()) {
                                        *logger << input_name
                                                << " is found from self common "
                                                   "parameter table"
                                                << std::endl;
                                        auto result_pair =
                                          variable_memory_cache_table_.emplace(
                                            input_name,
                                            memory_cache(found->second,
                                                         engine_));
                                        assert(
                                          result_pair.second &&
                                          "already same named variable exist");
                                        input_memory_cache_list.push_back(
                                          std::ref(result_pair.first->second));
                                        break;
                                    }
                                }

                                // search in common input table
                                {
                                    auto found =
                                      common_input_table.find(input_name);
                                    if(found != common_input_table.end()) {
                                        *logger << input_name
                                                << " is found from self common "
                                                   "input table"
                                                << std::endl;
                                        auto result_pair =
                                          variable_memory_cache_table_.emplace(
                                            input_name,
                                            memory_cache(found->second,
                                                         engine_));
                                        assert(
                                          result_pair.second &&
                                          "already same named variable exist");
                                        input_memory_cache_list.push_back(
                                          std::ref(result_pair.first->second));
                                        break;
                                    }
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
                                        *logger
                                          << input_name
                                          << " is found in other context's "
                                             "varibale table: "
                                          << context_pair.first << std::endl;
                                        procedure copy_proc; // copy from other
                                                             // context to array
                                        array arr;
                                        std::tie(copy_proc, arr) = *found;
                                        new_copy_procedure_list.push_back(
                                          copy_proc);
                                        auto result_pair =
                                          variable_memory_cache_table_.emplace(
                                            input_name,
                                            memory_cache(arr, engine_));
                                        assert(
                                          result_pair.second &&
                                          "already same named variable exist");
                                        input_memory_cache_list.push_back(
                                          std::ref(result_pair.first->second));
                                        is_found_from_other_context = true;
                                        break;
                                    }
                                }
                                assert(is_found_from_other_context);
                            } while(false);
                        }

                        std::vector<formatted_array>
                          output_formatted_array_list;
                        for(auto const& output_name : node.output_name_list) {
                            auto found =
                              required_output_table.find(output_name);
                            assert(
                              variable_memory_cache_table_.find(output_name) ==
                                variable_memory_cache_table_.end() &&
                              "variable have not already exist");
                            if(found == required_output_table.end()) {
                                // not required output
                                // add `any` format memory
                                auto arr =
                                  array(output_profile_table.at(output_name));
                                allocated_array_list_.push_back(arr);
                                output_formatted_array_list.push_back(
                                  formatted_array(mkldnn::memory::format::any,
                                                  arr));
                            } else {
                                // required output
                                output_formatted_array_list.push_back(
                                  formatted_array(
                                    ndims_to_data_memory_format(
                                      found->second.dims().size()),
                                    found->second));
                            }
                        }

                        auto found =
                          procedure_factory_table_.find(node.op_type);
                        if(found == procedure_factory_table_.end()) {
                            throw std::runtime_error("factory not found for: " +
                                                     node.op_type);
                        }
                        auto factory = found->second;
                        procedure_factory_return_type factory_return =
                          factory.operator()(node, input_memory_cache_list,
                                             output_formatted_array_list,
                                             engine_);
                        new_primitive_list = factory_return.primitives;
                        new_output_memory_cache_list =
                          factory_return.output_memory_cache_list;
                        new_named_temp_memory_cache_list =
                          factory_return.named_temp_memory_cache_list;
                    } catch(std::exception const& e) {
                        *logger << e.what() << std::endl;
                        break;
                    }

                    primitive_list.insert(primitive_list.end(),
                                          new_primitive_list.begin(),
                                          new_primitive_list.end());

                    // add copy procedures
                    procedure_list.insert(
                      procedure_list.end(),
                      std::make_move_iterator(new_copy_procedure_list.begin()),
                      std::make_move_iterator(new_copy_procedure_list.end()));

                    // update context
                    for(unsigned int i = 0; i < node.output_name_list.size();
                        ++i) {
                        variable_memory_cache_table_.emplace(
                          node.output_name_list.at(i),
                          new_output_memory_cache_list.at(i));
                    }
                    temp_memory_cache_table_.insert(
                      new_named_temp_memory_cache_list.begin(),
                      new_named_temp_memory_cache_list.end());
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
