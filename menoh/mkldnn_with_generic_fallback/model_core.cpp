#include <menoh/mkldnn_with_generic_fallback/model_core.hpp>

#include <algorithm>
#include <cassert>
#include <memory>

#include <menoh/mkldnn_with_generic_fallback/computation_node_factory.hpp>
#include <menoh/mkldnn_with_generic_fallback/context.hpp>

#include <menoh/mkldnn/utility.hpp>

#include <menoh/optional.hpp>
#include <menoh/utility.hpp>

#include <iostream>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {

        model_core::model_core(
          std::vector<std::pair<std::string, std::unique_ptr<context>>>
            context_list,
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data)
          : menoh_impl::model_core(),
            common_parameter_table_(
              model_data.parameter_name_and_array_list.begin(),
              model_data.parameter_name_and_array_list.end()),
            common_input_table_(input_table.begin(), input_table.end()),
            required_output_table_(output_table.begin(), output_table.end()),
            context_list_(std::move(context_list)) {

            auto graph = make_graph(model_data.node_list);

            for(decltype(graph.node_list().size()) current_index = 0;
                current_index < graph.node_list().size();) {
                auto const& node = graph.node_list().at(current_index);
                std::cout << "node: " << node.op_type << std::endl;
                std::cout << context_list_.size() << std::endl;

                // for each backend
                bool is_found = false;
                for(auto const& context_pair : context_list_) {
                    std::cout << "context: " << context_pair.first << std::endl;
                    auto const& context_name = context_pair.first;
                    auto context = context_pair.second.get();

                    // try to process nodes
                    optional<std::tuple<std::vector<procedure>, int>> result =
                      context->process_node_list(
                        context_name, current_index, graph.node_list(),
                        common_parameter_table_, common_input_table_,
                        required_output_table_, context_list_);

                    // if succeeded processing, add procedures into
                    // procedure_list
                    if(result) {
                        std::cout << "consumed "
                                  << std::get<1>(*result) - current_index
                                  << " nodes" << std::endl;
                        /*
                        std::cout << "find op "
                                  << graph.node_list().at(current_index).op_type
                                  << " in " << context_pair.first << std::endl;
                        */
                        std::vector<procedure> additional_procedure_list;
                        std::tie(additional_procedure_list, current_index) =
                          *result;
                        procedure_list_.insert(
                          procedure_list_.end(),
                          std::make_move_iterator(
                            additional_procedure_list.begin()),
                          std::make_move_iterator(
                            additional_procedure_list.end()));
                        is_found = true;
                        break;
                    } else {
                        std::cout << "failed to find op "
                                  << graph.node_list().at(current_index).op_type
                                  << " in " << context_pair.first << std::endl;
                    }
                }
                // if any context can not process the node
                if(!is_found) {
                    throw unsupported_operator(node.op_type);
                }
            }

            // erase useless procedures
            auto end_iter = std::remove_if(
              procedure_list_.begin(), procedure_list_.end(),
              [](auto const& e) { return !static_cast<bool>(e); });
            procedure_list_.erase(end_iter, procedure_list_.end());
        }

        void model_core::do_run() {
            for(auto const& procedure : procedure_list_) {
                procedure.operator()();
            }
        }

    } // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl
