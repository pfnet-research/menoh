#include <menoh/onnx.hpp>

#include <algorithm>
#include <exception>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <external/onnx/onnx/onnx.pb.h>

#include <menoh/array.hpp>
#include <menoh/dtype.hpp>
#include <menoh/exception.hpp>
#include <menoh/graph.hpp>
#include <menoh/model_data.hpp>
#include <menoh/node.hpp>

namespace menoh_impl {

    onnx::ModelProto load_onnx_model_proto(std::string const& filename) {
        namespace gpio = ::google::protobuf::io;

        std::ifstream ifs(filename, std::ios::binary);
        if(!ifs) {
            throw invalid_filename(filename);
        }
        gpio::IstreamInputStream iis(&ifs);
        gpio::CodedInputStream cis(&iis);
        cis.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                               std::numeric_limits<int>::max());
        onnx::ModelProto onnx_model;
        if(!onnx_model.ParseFromCodedStream(&cis)) {
            throw onnx_parse_error(filename);
        }
        return onnx_model;
    }

    auto tensor_proto_data_type_to_dtype(onnx::TensorProto_DataType tpdt) {
        if(tpdt == onnx::TensorProto_DataType_FLOAT) {
            return dtype_t::float_;
        }
        throw invalid_dtype(std::to_string(tpdt));
    }

    auto extract_parameter_name_set(onnx::GraphProto const& graph) {
        std::set<std::string> parameter_name_set;
        for(int i = 0; i < graph.initializer_size(); ++i) {
            auto& tensor = graph.initializer(i);
            parameter_name_set.insert(tensor.name());
        }
        return parameter_name_set;
    }

    auto
    extract_needed_input_name_set(std::vector<node> const& node_list,
                                  std::set<std::string> parameter_name_set) {
        std::set<std::string> input_name_set;
        for(auto const& node : node_list) {
            input_name_set.insert(node.input_name_list.begin(),
                                  node.input_name_list.end());
            parameter_name_set.insert(node.output_name_list.begin(),
                                      node.output_name_list.end());
        }
        std::set<std::string> needed_input_name_set;
        std::set_difference(
          input_name_set.begin(), input_name_set.end(),
          parameter_name_set.begin(), parameter_name_set.end(),
          std::inserter(needed_input_name_set, needed_input_name_set.end()));
        return needed_input_name_set;
    }

    auto extract_needed_parameter_name_set(
      std::vector<node> const& node_list,
      std::set<std::string> given_input_name_set) {
        std::set<std::string> input_name_set;
        for(auto const& node : node_list) {
            input_name_set.insert(node.input_name_list.begin(),
                                  node.input_name_list.end());
            given_input_name_set.insert(node.output_name_list.begin(),
                                        node.output_name_list.end());
        }
        std::set<std::string> needed_parameter_name_set;
        std::set_difference(input_name_set.begin(), input_name_set.end(),
                            given_input_name_set.begin(),
                            given_input_name_set.end(),
                            std::inserter(needed_parameter_name_set,
                                          needed_parameter_name_set.end()));
        return needed_parameter_name_set;
    }

    auto extract_parameter_name_and_array_list_from_onnx_graph(
      onnx::GraphProto& graph,
      std::vector<std::string> const& needed_parameter_name_list) {
        std::vector<std::pair<std::string, menoh_impl::array>>
          parameter_name_and_array_list;
        for(int i = 0; i < graph.initializer_size(); ++i) {
            auto& tensor = *graph.mutable_initializer(i);
            if(std::find(needed_parameter_name_list.begin(),
                         needed_parameter_name_list.end(),
                         tensor.name()) == needed_parameter_name_list.end()) {
                continue;
            }
            assert(tensor.has_data_type());

            std::vector<int> dims(tensor.dims().begin(), tensor.dims().end());
            auto total_size = std::accumulate(dims.begin(), dims.end(), 1,
                                              std::multiplies<int>());

            // FIXME workaround for Reshape-5
            if(tensor.data_type() == onnx::TensorProto_DataType_INT64) {
                parameter_name_and_array_list.push_back(
                  {tensor.name(),
                   menoh_impl::array(dtype_t::float_, std::move(dims))});
            }
            // end workaround

            dtype_t d = tensor_proto_data_type_to_dtype(tensor.data_type());

            std::shared_ptr<void> data;
            if(d == menoh_impl::dtype_t::float_) {
                using float_t =
                  menoh_impl::dtype_to_type_t<menoh_impl::dtype_t::float_>;
                // libc++ workaround
                // Below 2 lines are equal to `data =
                // std::unique_ptr<float_t[]>(new float_t[total_size]);`
                auto u = std::make_unique<float_t[]>(total_size);
                data = std::shared_ptr<void>(u.release(), u.get_deleter());
                // TODO other format: float_data
                assert(tensor.has_raw_data());
                assert(tensor.raw_data().length() ==
                       static_cast<decltype(tensor.raw_data().length())>(
                         total_size * 4));
                std::copy(tensor.raw_data().begin(), tensor.raw_data().end(),
                          static_cast<char*>(data.get()));
                delete tensor.release_raw_data();
            } else {
                throw invalid_dtype(std::to_string(tensor.data_type()));
            }
            parameter_name_and_array_list.push_back(
              {tensor.name(),
               menoh_impl::array(d, std::move(dims), std::move(data))});
        }
        return parameter_name_and_array_list;
    }

    auto extract_node_list_from_onnx_graph(onnx::GraphProto const& graph) {
        std::vector<node> node_list;
        for(auto const& onnx_node : graph.node()) {
            std::unordered_map<std::string, attribute> attribute_table;
            for(auto const& attr : onnx_node.attribute()) {
                if(attr.has_i()) {
                    attribute_table.insert(
                      {attr.name(), static_cast<int>(attr.i())}); // TODO int64
                } else if(attr.has_f()) {
                    attribute_table.insert({attr.name(), attr.f()});
                } else if(attr.ints_size()) {
                    attribute_table.insert(
                      {attr.name(), std::vector<int>(attr.ints().begin(),
                                                     attr.ints().end())});
                } else if(attr.floats_size()) {
                    attribute_table.insert(
                      {attr.name(), std::vector<float>(attr.floats().begin(),
                                                       attr.floats().end())});
                } else {
                    throw invalid_attribute_type(
                      attr.name(),
                      AttributeProto_AttributeType_Name(attr.type()));
                }
            }
            menoh_impl::node n{
              onnx_node.op_type(),
              std::vector<std::string>(onnx_node.input().begin(),
                                       onnx_node.input().end()),
              std::vector<std::string>(onnx_node.output().begin(),
                                       onnx_node.output().end()),
              attribute_table};
            node_list.push_back(n);
        }
        return node_list;
    }

    model_data load_onnx(std::string const& filename) {
        auto onnx_model = load_onnx_model_proto(filename);

        // onnx opset version check
        if(onnx_model.opset_import_size() != 0) {
            int version = onnx_model.opset_import(0).version();
            if(MENOH_SUPPORTED_ONNX_OPSET_VERSION < version) {
                throw unsupported_onnx_opset_version(
                  filename, version, MENOH_SUPPORTED_ONNX_OPSET_VERSION);
            }
        }

        auto node_list = extract_node_list_from_onnx_graph(onnx_model.graph());

        trim_dropout(node_list);
        trim_reshape(node_list);

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(node_list.begin(), node_list.end(), g);

        std::vector<std::string> all_parameter_name_list;
        all_parameter_name_list.reserve(
          onnx_model.graph().initializer().size());
        std::transform(onnx_model.graph().initializer().begin(),
                       onnx_model.graph().initializer().end(),
                       std::back_inserter(all_parameter_name_list),
                       [](auto const& tensor) { return tensor.name(); });

        auto all_input_name_set = extract_all_input_name_set(node_list);

        std::vector<std::string> model_parameter_name_list;
        std::sort(all_parameter_name_list.begin(),
                  all_parameter_name_list.end());
        std::set_intersection(
          all_parameter_name_list.begin(), all_parameter_name_list.end(),
          all_input_name_set.begin(), all_input_name_set.end(),
          std::back_inserter(model_parameter_name_list));

        auto parameter_table =
          extract_parameter_name_and_array_list_from_onnx_graph(
            *onnx_model.mutable_graph(), model_parameter_name_list);
        return model_data{node_list, parameter_table};
    }

    std::vector<std::string>
    extract_model_input_name_list(menoh_impl::model_data const& model_data) {
        auto all_input_name_set =
          extract_all_input_name_set(model_data.node_list);
        auto all_output_name_set =
          extract_all_output_name_set(model_data.node_list);

        std::vector<std::string> parameter_name_and_all_output_name_list;
        parameter_name_and_all_output_name_list.reserve(
          model_data.parameter_name_and_array_list.size() +
          all_output_name_set.size());
        std::transform(
          model_data.parameter_name_and_array_list.begin(),
          model_data.parameter_name_and_array_list.end(),
          std::back_inserter(parameter_name_and_all_output_name_list),
          [](auto const& p) { return p.first; });
        parameter_name_and_all_output_name_list.insert(
          parameter_name_and_all_output_name_list.end(),
          all_output_name_set.begin(), all_output_name_set.end());

        std::vector<std::string> model_input_name_list;
        std::sort(parameter_name_and_all_output_name_list.begin(),
                  parameter_name_and_all_output_name_list.end());
        std::set_difference(all_input_name_set.begin(),
                            all_input_name_set.end(),
                            parameter_name_and_all_output_name_list.begin(),
                            parameter_name_and_all_output_name_list.end(),
                            std::back_inserter(model_input_name_list));

        return model_input_name_list;
    }

    std::vector<std::string>
    extract_model_output_name_list(menoh_impl::model_data const& model_data) {
        auto all_input_name_set =
          extract_all_input_name_set(model_data.node_list);
        auto all_output_name_set =
          extract_all_output_name_set(model_data.node_list);

        std::vector<std::string> model_output_name_list;
        std::set_difference(
          all_output_name_set.begin(), all_output_name_set.end(),
          all_input_name_set.begin(), all_input_name_set.end(),
          std::back_inserter(model_output_name_list));

        return model_output_name_list;
    }

} // namespace menoh_impl
