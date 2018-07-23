#include <menoh/mkldnn/model_core.hpp>

#include <functional>
#include <iterator>
#include <tuple>

#include <menoh/exception.hpp>
#include <menoh/graph.hpp>
#include <menoh/json.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>
#include <menoh/utility.hpp>

#include <menoh/mkldnn/operator.hpp>
#include <menoh/mkldnn/primitive_factory_return_type.hpp>
#include <menoh/mkldnn/utility.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        using inplace_primitive_factory =
          std::function<primitive_factory_return_type(
            node,
            int,                      // node_index
            std::vector<node> const&, // node_list
            std::unordered_map<std::string,
                               array> const&, // parameter table
            std::unordered_map<std::string,
                               mkldnn::memory> const&, // variable memory table
            std::unordered_map<std::string,
                               array> const&, // required output table
            mkldnn::engine const&)>;

        using primitive_factory = std::function<primitive_factory_return_type(
          node,
          std::unordered_map<std::string,
                             array> const&, // parameter table
          std::unordered_map<std::string,
                             mkldnn::memory> const&, // variable memory table
          std::unordered_map<std::string,
                             array> const&, // required output table
          mkldnn::engine const&)>;

        auto make_nets_core(
          menoh_impl::graph const& graph,
          std::unordered_map<std::string, array> const& parameter_table,
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          std::unordered_map<std::string, inplace_primitive_factory>
            inplace_primitive_factory_table,
          std::unordered_map<std::string, primitive_factory>
            primitive_factory_table,
          mkldnn::engine const& engine) {
            std::vector<mkldnn::primitive> nets;
            std::unordered_map<std::string, mkldnn::memory>
              variable_memory_table;
            for(auto const& name_and_arr_pair : input_table) {
                std::string name;
                array arr;
                std::tie(name, arr) = name_and_arr_pair;
                mkldnn::memory::format format;
                if(arr.dims().size() == 2) {
                    format = mkldnn::memory::format::nc;
                } else if(arr.dims().size() == 4) {
                    format = mkldnn::memory::format::nchw;
                } else {
                    assert("invalid input dims size");
                }
                auto mem = array_to_memory(arr, format, engine);
                variable_memory_table.insert({name, mem});
            }
            std::vector<mkldnn::memory> temp_memory_list;
            std::vector<array> owned_array_list;
            std::set<std::string> variable_name_set;
            std::vector<std::string> output_name_sorted_list;
            {
                std::transform(output_table.begin(), output_table.end(),
                               std::back_inserter(output_name_sorted_list),
                               [](auto const& e) { return e.first; });
                std::sort(output_name_sorted_list.begin(),
                          output_name_sorted_list.end());
            }
            for(decltype(graph.node_list().size()) i = 0;
                i < graph.node_list().size(); ++i) {
                auto const& node = graph.node_list().at(i);
                try {
                    auto inplace_primitive_factory_pair_iter =
                      inplace_primitive_factory_table.find(node.op_type);
                    auto primitive_factory_pair_iter =
                      primitive_factory_table.find(node.op_type);
                    if(inplace_primitive_factory_pair_iter ==
                         inplace_primitive_factory_table.end() &&
                       primitive_factory_pair_iter ==
                         primitive_factory_table.end()) {
                        throw unsupported_operator(node.op_type);
                    }

                    // Call a primitive factory
                    std::vector<mkldnn::primitive> net;
                    std::unordered_map<std::string, mkldnn::memory>
                      new_output_memory_table;
                    std::vector<mkldnn::memory> new_temp_memory_list;
                    std::vector<array> new_owned_array_list;
                    if(inplace_primitive_factory_pair_iter !=
                       inplace_primitive_factory_table.end()) {
                        std::tie(net, new_output_memory_table,
                                 new_temp_memory_list, new_owned_array_list) =
                          inplace_primitive_factory_pair_iter->second.
                          operator()(node, i, graph.node_list(),
                                     parameter_table, variable_memory_table,
                                     output_table, engine);
                    } else {
                        assert(primitive_factory_pair_iter !=
                               primitive_factory_table.end());
                        std::tie(net, new_output_memory_table,
                                 new_temp_memory_list, new_owned_array_list) =
                          primitive_factory_pair_iter->second.operator()(
                            node, parameter_table, variable_memory_table,
                            output_table, engine);
                    }

                    nets.insert(nets.end(), net.begin(), net.end());
                    variable_memory_table.insert(
                      new_output_memory_table.begin(),
                      new_output_memory_table.end());
                    temp_memory_list.insert(temp_memory_list.end(),
                                            new_temp_memory_list.begin(),
                                            new_temp_memory_list.end());
                    owned_array_list.insert(owned_array_list.end(),
                                            new_owned_array_list.begin(),
                                            new_owned_array_list.end());

                    // Check weather all required output is emitted
                    std::transform(
                      new_output_memory_table.begin(),
                      new_output_memory_table.end(),
                      std::inserter(variable_name_set, variable_name_set.end()),
                      [](auto const& e) { return e.first; });
                    std::vector<std::string> diffs;
                    std::set_difference(
                      output_name_sorted_list.begin(),
                      output_name_sorted_list.end(), variable_name_set.begin(),
                      variable_name_set.end(), std::back_inserter(diffs));
                    if(diffs.empty()) {
                        break;
                    }
                } catch(mkldnn::error const& e) {
                    throw failed_to_configure_operator(
                      node.op_type, node.output_name_list.at(0),
                      std::string("status: ") + std::to_string(e.status) +
                        ", message: " + e.message);
                }
            }
            return std::make_tuple(nets, variable_memory_table,
                                   temp_memory_list, owned_array_list);
        }

        auto
        make_nets(std::unordered_map<std::string, array> const& input_table,
                  std::unordered_map<std::string, array> const& output_table,
                  menoh_impl::model_data const& model_data,
                  mkldnn::engine const& engine) {
            std::unordered_map<std::string, inplace_primitive_factory>
              inplace_primitive_factory_table;
            inplace_primitive_factory_table.insert({"Abs", make_abs_primitive});
            inplace_primitive_factory_table.insert({"Elu", make_elu_primitive});
            inplace_primitive_factory_table.insert(
              {"LeakyRelu", make_leaky_relu_primitive});
            inplace_primitive_factory_table.insert(
              {"Relu", make_relu_primitive});
            inplace_primitive_factory_table.insert(
              {"Sqrt", make_sqrt_primitive});
            inplace_primitive_factory_table.insert(
              {"Tanh", make_tanh_primitive});

            std::unordered_map<std::string, primitive_factory>
              primitive_factory_table;
            primitive_factory_table.insert(
              {"AveragePool", make_average_pool_primitive});
            primitive_factory_table.insert({"Add", make_add_primitive});
            primitive_factory_table.insert(
              {"BatchNormalization", make_batch_norm_primitive});
            primitive_factory_table.insert({"Concat", make_concat_primitive});
            primitive_factory_table.insert({"Conv", make_conv_primitive});
            primitive_factory_table.insert(
              {"ConvTranspose", make_conv_transpose_primitive});
            primitive_factory_table.insert({"FC", make_fc_primitive});
            primitive_factory_table.insert({"Gemm", make_gemm_primitive});
            primitive_factory_table.insert(
              {"GlobalAveragePool", make_global_average_pool_primitive});
            primitive_factory_table.insert(
              {"GlobalMaxPool", make_global_max_pool_primitive});
            primitive_factory_table.insert({"LRN", make_lrn_primitive});
            primitive_factory_table.insert(
              {"MaxPool", make_max_pool_primitive});
            primitive_factory_table.insert({"Softmax", make_softmax_primitive});
            primitive_factory_table.insert({"Sum", make_sum_primitive});
            // TODO other primitives

            auto graph = make_graph(model_data.node_list);
            auto parameter_table = std::unordered_map<std::string, array>(
              model_data.parameter_name_and_array_list.begin(),
              model_data.parameter_name_and_array_list.end());
            return make_nets_core(graph, parameter_table, input_table,
                                  output_table, inplace_primitive_factory_table,
                                  primitive_factory_table, engine);
        }

        model_core::model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data,
          mkldnn::engine const& engine)
          : engine_(engine) {
            std::tie(nets_, variable_memory_table_, temp_memory_list_,
                     owned_array_list_) =
              make_nets(input_table, output_table, model_data, engine_);
        }

        void model_core::do_run() {
            try {
                mkldnn::stream(mkldnn::stream::kind::eager)
                  .submit(nets_)
                  .wait();
            } catch(mkldnn::error const& e) {
                throw backend_error("mkldnn", std::string("status: ") +
                                                std::to_string(e.status) +
                                                ", message: " + e.message);
            }
        }

        model_core make_model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data,
          backend_config const& config) {
            try {
                int cpu_id = 0;
                if(!config.empty()) {
                    auto c = nlohmann::json::parse(config);
                    if(c.find("cpu_id") != c.end()) {
                        cpu_id = c["cpu_id"].get<int>();
                    }
                }
                mkldnn::engine engine(mkldnn::engine::cpu, cpu_id);
                return model_core(input_table, output_table, model_data,
                                  engine);
            } catch(nlohmann::json::parse_error const& e) {
                throw json_parse_error(e.what());
            } catch(mkldnn::error const& e) {
                throw backend_error("mkldnn", std::string("status: ") +
                                                std::to_string(e.status) +
                                                ", message: " + e.message);
            }
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
