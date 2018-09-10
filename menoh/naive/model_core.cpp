#include <menoh/naive/model_core.hpp>

#include <algorithm>
#include <cassert>

#include <menoh/naive/computation_node.hpp>
#include <menoh/naive/computation_node_factory.hpp>

#include <menoh/naive/operator/elu.hpp>
#include <menoh/naive/operator/leaky_relu.hpp>
#include <menoh/naive/operator/relu.hpp>
#include <menoh/naive/operator/softmax.hpp>
#include <menoh/naive/operator/tanh.hpp>

#include <menoh/naive/operator/concat.hpp>

#include <menoh/naive/operator/gemm.hpp>

#include <menoh/naive/operator/add.hpp>
#include <menoh/naive/operator/sum.hpp>

#include <menoh/naive/operator/average_pool.hpp>
#include <menoh/naive/operator/max_pool.hpp>

namespace menoh_impl {
    namespace naive_backend {

        model_core::model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data) {

            variable_table_.insert(input_table.begin(), input_table.end());
            variable_table_.insert(
              model_data.parameter_name_and_array_list.begin(),
              model_data.parameter_name_and_array_list.end());
            variable_table_.insert(output_table.begin(), output_table.end());

            auto graph = make_graph(model_data.node_list);
            auto parameter_table = std::unordered_map<std::string, array>(
              model_data.parameter_name_and_array_list.begin(),
              model_data.parameter_name_and_array_list.end());

            std::unordered_map<std::string, computation_node_factory>
              computation_node_factory_table = {
                {"Add", make_add},          {"AveragePool", make_average_pool},
                {"Concat", make_concat},    {"Elu", make_elu},
                {"Gemm", make_gemm},        {"LeakyRelu", make_leaky_relu},
                {"MaxPool", make_max_pool}, {"Relu", make_relu},
                {"Softmax", make_softmax},  {"Sum", make_sum},
                {"Tanh", make_tanh}};

            for(std::size_t i = 0; i < graph.node_list().size(); ++i) {
                auto const& node = graph.node_list().at(i);

                // check if supported operator
                auto found_factory =
                  computation_node_factory_table.find(node.op_type);
                if(found_factory == computation_node_factory_table.end()) {
                    throw unsupported_operator(node.op_type);
                }

                std::function<void()> computation_node;
                std::vector<std::pair<std::string, array>>
                  new_variable_named_list;
                std::tie(computation_node, new_variable_named_list) =
                  found_factory->second.operator()(i, graph.node_list(),
                                                   variable_table_);

                computation_node_list_.push_back(computation_node);

                {
                    // uniqueness check
                    assert(std::unique(new_variable_named_list.begin(),
                                       new_variable_named_list.end(),
                                       [](auto const& a, auto const& b) {
                                           return a.first == b.first;
                                       }) == new_variable_named_list.end());
                }
                variable_table_.insert(new_variable_named_list.begin(),
                                       new_variable_named_list.end());
            }
        }

        void model_core::do_run() {
            for(auto const& computation_node : computation_node_list_) {
                computation_node.operator()();
            }
        }

    } // namespace naive_backend
} // namespace menoh_impl
