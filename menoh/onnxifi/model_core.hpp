#ifndef MENOH_IMPL_ONNXIFI_BACKEND_MODEL_CORE_HPP
#define MENOH_IMPL_ONNXIFI_BACKEND_MODEL_CORE_HPP

#include <list>
#include <string>
#include <vector>

#include <onnx/onnx.pb.h>
#include <onnx/onnxifi.h>
#include <onnx/onnxifi_loader.h>

#include <fmt/format.h>

#include <menoh/backend_config.hpp>
#include <menoh/dtype.hpp>
#include <menoh/graph.hpp>
#include <menoh/json.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>

namespace menoh_impl {
    namespace onnxifi_backend {

        template <typename... Args>
        void debug_print(const char* format, const Args&... args) {
#define MENOH_ONNXIFI_BACKEND_DEBUG
#ifdef MENOH_ONNXIFI_BACKEND_DEBUG
            [format](fmt::format_args args) {
                fmt::vprint(format, args);
            }(fmt::make_format_args(args...));
#endif // MENOH_ONNXIFI_BACKEND_DEBUG
        }

        class backend_ids_deleter {
        public:
            explicit backend_ids_deleter(onnxifi_library const& lib,
                                         size_t size)
              : lib_(lib), size_(size) {}

            void operator()(onnxBackendID* p) {
                for(size_t i = 0; i < size_; ++i) {
                    lib_.onnxReleaseBackendID(*(p + i));
                }
                delete[] p;
            }

        private:
            onnxifi_library lib_;
            size_t size_;
        };

        using backend_ids_ptr =
          std::unique_ptr<onnxBackendID[], backend_ids_deleter>;
        auto get_backend_id_list(onnxifi_library const& lib) {
            size_t backends_num = -1;
            lib.onnxGetBackendIDs(nullptr, &backends_num);
            backend_ids_ptr backend_id_list(
              new onnxBackendID[backends_num],
              backend_ids_deleter(lib, backends_num));
            lib.onnxGetBackendIDs(backend_id_list.get(), &backends_num);
            return backend_id_list;
        }

        class backend_deleter {
        public:
            backend_deleter(onnxifi_library lib) : lib_(lib) {}

            void operator()(onnxBackendID* p) {
                lib_.onnxReleaseBackend(*p);
                delete p;
            }

        private:
            onnxifi_library lib_;
        };

        using backend_ptr = std::unique_ptr<onnxBackend, backend_deleter>;
        auto init_backend(onnxifi_library const& lib, onnxBackendID backend_id,
                          const uint64_t* properties) {
            std::unique_ptr<onnxBackend, backend_deleter> backend(
              new onnxBackend(), backend_deleter(lib));
            lib.onnxInitBackend(backend_id, properties, backend.get());
            return backend;
        }

        auto to_onnx_node_proto(node const& node) {
            onnx::NodeProto node_proto;
            node_proto.set_op_type(node.op_type);
            for(auto const& input_name : node.input_name_list) {
                node_proto.add_input(input_name);
            }
            for(auto const& output_name : node.output_name_list) {
                node_proto.add_output(output_name);
            }
            for(auto const& name_and_attribute : node.attribute_table) {
                auto const& name = name_and_attribute.first;
                auto const& attr = name_and_attribute.second;
                auto attr_proto = node_proto.add_attribute();
                attr_proto->set_name(name);

                const int* i;
                const float* f;
                const std::vector<int>* ints;
                const std::vector<float>* floats;
                if(i = get_if<int>(&attr)) {
                    attr_proto->set_i(static_cast<int64_t>(*i));
                } else if(f = get_if<float>(&attr)) {
                    attr_proto->set_f(static_cast<float>(*f));
                } else if(ints = get_if<std::vector<int>>(&attr)) {
                    for(auto i : *ints) {
                        attr_proto->add_ints(static_cast<int64_t>(i));
                    }
                } else if(floats = get_if<std::vector<float>>(&attr)) {
                    for(auto f : *floats) {
                        attr_proto->add_floats(f);
                    }
                }
            }
            return node_proto;
        }

        auto menoh_dtype_to_onnx_dtype(dtype_t dtype) {
            if(dtype == dtype_t::float_) {
                fmt::format("dtype is {}", "float");
                return onnx::TensorProto::FLOAT;
            }
            throw std::runtime_error("not supported dtype");
        }

        auto to_onnx_model_ptoro_without_initializer(
          model_data const& model_data,
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table) {
            onnx::ModelProto model_proto;
            model_proto.set_ir_version(model_data.ir_version);
            auto opset_import = model_proto.add_opset_import();
            opset_import->set_version(model_data.opset_version);
            onnx::GraphProto* graph_proto = model_proto.mutable_graph();
            auto g = make_graph(model_data.node_list);

            auto set_value_info = [](std::string const& name, array const& arr,
                                     onnx::ValueInfoProto* value_info_proto) {
                value_info_proto->set_name(name);
                auto type_proto = value_info_proto->mutable_type();
                auto type_proto_tensor = type_proto->mutable_tensor_type();
                type_proto_tensor->set_elem_type(
                  menoh_dtype_to_onnx_dtype(arr.dtype()));
                auto tensor_shape_proto = type_proto_tensor->mutable_shape();
                for(auto d : arr.dims()) {
                    auto tensor_shape_proto_dimension =
                      tensor_shape_proto->add_dim();
                    tensor_shape_proto_dimension->set_dim_value(
                      static_cast<int64_t>(d));
                }
            };
            for(auto const& name_and_array : input_table) {
                auto const& name = name_and_array.first;
                auto const& arr = name_and_array.second;
                auto value_info_proto = graph_proto->add_input();
                set_value_info(name, arr, value_info_proto);
            }
            for(auto const& name_and_array : output_table) {
                auto const& name = name_and_array.first;
                auto const& arr = name_and_array.second;
                auto value_info_proto = graph_proto->add_output();
                set_value_info(name, arr, value_info_proto);
            }
            for(auto const& node : g.node_list()) {
                debug_print("{}: {} -> {}\n", node.op_type,
                            node.input_name_list.front(),
                            node.output_name_list.front());
                *graph_proto->add_node() = to_onnx_node_proto(node);
            }
            return model_proto;
        }

        class graph_deleter {
        public:
            graph_deleter(onnxifi_library lib) : lib_(lib) {}

            void operator()(onnxBackendID* p) {
                lib_.onnxReleaseGraph(*p);
                delete p;
            }

        private:
            onnxifi_library lib_;
        };

        using graph_ptr = std::unique_ptr<onnxGraph, graph_deleter>;
        auto
        init_graph(onnxifi_library const& lib, backend_ptr const& backend,
                   const uint64_t* properties,
                   std::string const& serialized_model,
                   std::vector<onnxTensorDescriptorV1> const& weight_descs) {
            graph_ptr graph(new onnxGraph(), graph_deleter(lib));
            lib.onnxInitGraph(
              *backend, properties, serialized_model.size(),
              static_cast<void*>(const_cast<char*>(serialized_model.c_str())),
              weight_descs.size(), weight_descs.data(), graph.get());
            return graph;
        }

        auto
        set_graph_io(onnxifi_library const& lib, graph_ptr const& graph,
                     std::vector<onnxTensorDescriptorV1> const& input_descs,
                     std::vector<onnxTensorDescriptorV1> const& output_descs) {
            auto res =
              lib.onnxSetGraphIO(*graph, input_descs.size(), input_descs.data(),
                                 output_descs.size(), output_descs.data());
            debug_print("set_graph_io result: {}\n", res);
        }

        class model_core final : public menoh_impl::model_core {
        public:
            model_core(
              onnxifi_library const& onnxifi_lib,
              std::unordered_map<std::string, array> const& input_table,
              std::unordered_map<std::string, array> const& output_table,
              menoh_impl::model_data const& model_data)
              : onnxifi_lib_(onnxifi_lib) {

                backend_ids_ = get_backend_id_list(onnxifi_lib);
                const uint64_t backend_properties[] = {
                  ONNXIFI_BACKEND_PROPERTY_NONE};
                backend_ = init_backend(
                  onnxifi_lib, backend_ids_[0],
                  backend_properties); // TODO specify-able backend id
                weight_descs_ =
                  make_tensor_descs(model_data.parameter_name_and_array_list);
                std::string serialized_model;
                to_onnx_model_ptoro_without_initializer(model_data, input_table,
                                                        output_table)
                  .SerializeToString(&serialized_model);
                const uint64_t* aux_properties_list = nullptr;
                graph_ = init_graph(onnxifi_lib_, backend_, aux_properties_list,
                                    serialized_model, weight_descs_);
                input_descs_ = make_tensor_descs(input_table);
                output_descs_ = make_tensor_descs(output_table);
                set_graph_io(onnxifi_lib_, graph_, input_descs_, output_descs_);
            }

        private:
            virtual void do_run() override {
                onnxMemoryFenceV1 input_fence;
                input_fence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
                input_fence.type = ONNXIFI_SYNCHRONIZATION_EVENT;
                onnxifi_lib_.onnxInitEvent(*backend_, &input_fence.event);
                onnxMemoryFenceV1 output_fence;
                output_fence.tag = ONNXIFI_TAG_MEMORY_FENCE_V1;
                output_fence.type = ONNXIFI_SYNCHRONIZATION_EVENT;

                onnxifi_lib_.onnxSignalEvent(input_fence.event);
                onnxifi_lib_.onnxRunGraph(*graph_, &input_fence, &output_fence);
                onnxifi_lib_.onnxWaitEvent(output_fence.event);

                onnxifi_lib_.onnxReleaseEvent(input_fence.event);
                onnxifi_lib_.onnxReleaseEvent(output_fence.event);
            }

        private:
            template <class NameAndArrayList>
            std::vector<onnxTensorDescriptorV1>
            make_tensor_descs(NameAndArrayList const& name_and_array_list) {
                std::vector<onnxTensorDescriptorV1> tensor_descs;
                for(auto& p : name_and_array_list) {
                    name_buffer_list_.push_back(p.first);
                    array_buffer_list_.push_back(p.second);
                    auto& arr = array_buffer_list_.back();
                    tensor_shape_buffer_list_.push_back(std::vector<uint64_t>(
                      arr.dims().begin(), arr.dims().end()));
                    tensor_descs.push_back(onnxTensorDescriptorV1{
                      ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1,
                      name_buffer_list_.back().c_str(),
                      menoh_dtype_to_onnx_dtype(arr.dtype()),
                      ONNXIFI_MEMORY_TYPE_CPU,
                      static_cast<uint32_t>(arr.dims().size()),
                      tensor_shape_buffer_list_.back().data(),
                      reinterpret_cast<onnxPointer>(arr.data())});
                }
                return tensor_descs;
            }
            std::list<std::string> name_buffer_list_;
            std::vector<array> array_buffer_list_;
            std::list<std::vector<uint64_t>> tensor_shape_buffer_list_;

        private:
            onnxifi_library onnxifi_lib_;
            backend_ids_ptr backend_ids_{nullptr,
                                         backend_ids_deleter(onnxifi_lib_, 0)};
            backend_ptr backend_{nullptr, backend_deleter(onnxifi_lib_)};
            graph_ptr graph_{nullptr, graph_deleter(onnxifi_lib_)};
            std::vector<onnxTensorDescriptorV1> weight_descs_;
            std::vector<onnxTensorDescriptorV1> input_descs_;
            std::vector<onnxTensorDescriptorV1> output_descs_;
        };

        model_core make_model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          std::unordered_map<std::string, array_profile> const&
            output_profile_table,
          menoh_impl::model_data const& model_data,
          backend_config const& config = backend_config()) {
            auto c = nlohmann::json::parse(config);
            auto library_path = c.find("library_path") == c.end()
                                  ? ""
                                  : c["library_path"].get<std::string>();
            debug_print("library_path: {}\n", library_path);
            onnxifi_library onnxifi_lib;
            if(!onnxifi_load(ONNXIFI_LOADER_FLAG_VERSION_1_0,
                             library_path.empty() ? nullptr
                                                  : library_path.c_str(),
                             &onnxifi_lib)) {
                throw std::runtime_error(
                  fmt::format("onnxifi initialize failed: {}", library_path)
                    .c_str());
            }
            return model_core(onnxifi_lib, input_table, output_table,
                              model_data);
        }

    } // namespace onnxifi_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_ONNXIFI_BACKEND_MODEL_CORE_HPP
