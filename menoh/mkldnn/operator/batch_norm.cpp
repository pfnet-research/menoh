#include <menoh/mkldnn/operator/batch_norm.hpp>

#include <tuple>

#include <menoh/utility.hpp>
#include <menoh/model_core.hpp>

#include <menoh/mkldnn/operator/common.hpp>
#include <menoh/mkldnn/utility.hpp>

namespace menoh_impl {
    namespace mkldnn_backend {

        primitive_factory_return_type make_batch_norm_primitive(
          menoh_impl::node const& node,
          std::unordered_map<std::string, array> const& parameter_table,
          std::unordered_map<std::string, mkldnn::memory> const&
            variable_memory_table,
          std::unordered_map<std::string, array> const& required_output_table,
          mkldnn::engine const& engine) {

            std::vector<mkldnn::primitive> net;
            std::unordered_map<std::string, mkldnn::memory> output_memory_table;
            std::unordered_map<std::string, array> output_table;
            std::vector<mkldnn::memory> temp_memory_list;
            std::vector<array> owned_array_list;

            auto epsilon = optional_attribute_float(node, "epsilon", 1e-5f);
            auto spatial =
              static_cast<bool>(optional_attribute_int(node, "spatial", 1));
            if(!spatial) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "only spatial one is supported");
            }

            auto const& input_memory =
              find_value(variable_memory_table, node.input_name_list.at(0));

            auto const& scale_arr =
              find_value(parameter_table, node.input_name_list.at(1));
            auto const& b_arr =
              find_value(parameter_table, node.input_name_list.at(2));
            std::vector<int> weights_dims{2};
            weights_dims.insert(weights_dims.end(), scale_arr.dims().begin(),
                                scale_arr.dims().end());
            mkldnn::memory weights_memory(
              {{{weights_dims},
                dtype_to_mkldnn_memory_data_type(scale_arr.dtype()),
                mkldnn::memory::format::nc},
               engine});
            assert(scale_arr.dtype() == dtype_t::float_);
            std::copy(fbegin(scale_arr), fend(scale_arr),
                      static_cast<float*>(weights_memory.get_data_handle()));
            std::copy(fbegin(b_arr), fend(b_arr),
                      static_cast<float*>(weights_memory.get_data_handle()) +
                        total_size(scale_arr));
            temp_memory_list.push_back(weights_memory);

            auto mean_memory = array_to_memory_and_deal_ownership(
              find_value(parameter_table, node.input_name_list.at(3)),
              mkldnn::memory::format::x, engine, temp_memory_list,
              owned_array_list);
            auto var_memory = array_to_memory_and_deal_ownership(
              find_value(parameter_table, node.input_name_list.at(4)),
              mkldnn::memory::format::x, engine, temp_memory_list,
              owned_array_list);

            auto input_dims = extract_dims(input_memory);

            auto c = input_dims[1];
            auto mean_dims = extract_dims(mean_memory);
            auto var_dims = extract_dims(var_memory);
            assert(scale_arr.dims()[0] == c && b_arr.dims()[0] == c &&
                   mean_dims[0] == c && var_dims[0] == c);

            auto const& output_name = node.output_name_list.at(0);

            mkldnn::batch_normalization_forward::desc bn_desc(
              mkldnn::prop_kind::forward_inference,
              input_memory.get_primitive_desc().desc(), epsilon,
              mkldnn::use_global_stats | mkldnn::use_scale_shift);
            auto bn_pd = mkldnn::batch_normalization_forward::primitive_desc(
              bn_desc, engine);

            assert(
              mkldnn::memory::primitive_desc(bn_pd.mean_primitive_desc()) ==
              mean_memory.get_primitive_desc());
            assert(
              mkldnn::memory::primitive_desc(bn_pd.variance_primitive_desc()) ==
              var_memory.get_primitive_desc());
            assert(
              mkldnn::memory::primitive_desc(bn_pd.weights_primitive_desc()) ==
              weights_memory.get_primitive_desc());

            manage_output_memory(
              net, output_name, mkldnn::memory::format::nchw,
              bn_pd.dst_primitive_desc(), output_memory_table,
              required_output_table, temp_memory_list, engine,
              [&net, &input_memory, &weights_memory, &mean_memory, &var_memory,
               &bn_pd](mkldnn::memory& op_output_memory) {
                  net.push_back(mkldnn::batch_normalization_forward(
                    bn_pd, static_cast<mkldnn::primitive::at>(input_memory),
                    static_cast<mkldnn::primitive::at>(mean_memory),
                    static_cast<mkldnn::primitive::at>(var_memory),
                    static_cast<mkldnn::primitive::at>(weights_memory),
                    op_output_memory));
              });

            return std::make_tuple(net, output_memory_table, temp_memory_list,
                                   owned_array_list);
        }

    } // namespace mkldnn_backend
} // namespace menoh_impl
