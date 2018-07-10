#include "../external/cmdline.h"
#include <external/onnx/onnx/onnx.pb.h>

namespace menoh_impl {
    onnx::ModelProto load_onnx_model_proto(std::string const& filename);
}

int main(int argc, char** argv) {
    if(argc == 1) {
        std::cout << "please set ONNX file path" << std::endl;
        return 0;
    }
    auto onnx_model_path = argv[1];
    auto onnx_model = menoh_impl::load_onnx_model_proto(onnx_model_path);

    std::cout << "ONNX version is " << onnx_model.ir_version() << std::endl;
    std::cout << "opset_import_size is " << onnx_model.opset_import_size() << std::endl;
    for(int i = 0; i < onnx_model.opset_import_size(); ++i) {
        std::cout << onnx_model.opset_import(i).version() << " ";
    }
    std::cout << "\n";
    std::cout << "domain is " << onnx_model.domain() << std::endl;
    std::cout << "model version is " << onnx_model.model_version() << std::endl;
    std::cout << "producer name is " << onnx_model.producer_name() << std::endl;
    std::cout << "producer version is " << onnx_model.producer_version() << std::endl;

    auto const& graph = onnx_model.graph();

    std::cout << "parameter list\n";
    for(auto const& tensor : graph.initializer()) {
        std::cout << "name: " << tensor.name();
        std::cout << " dtype: " << TensorProto_DataType_Name(tensor.data_type());
        std::cout << " dims: ";
        for(int j = 0; j < tensor.dims_size(); ++j) {
            std::cout << tensor.dims(j) << " ";
        }
        if(tensor.has_raw_data() && tensor.data_type() == onnx::TensorProto_DataType_FLOAT) {
            std::cout << " values: ";
            auto f = static_cast<float const*>(static_cast<void const*>(tensor.raw_data().c_str()));
            std::vector<float> floats;
            std::copy(f, f+tensor.raw_data().length()/sizeof(float), std::back_inserter(floats));
            auto max_value = *std::max_element(floats.begin(), floats.end());
            auto min_value = *std::min_element(floats.begin(), floats.end());
            std::cout << "min_value: " << min_value << " ";
            std::cout << "max_value: " << max_value << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "node list\n";
    std::cout << "node num is " << graph.node_size() << std::endl;
    for(int i = 0; i < graph.node_size(); ++i) {
        auto node = graph.node(i);
        std::cout << i << ":" << node.op_type() << std::endl;
        for(int j = 0; j < node.input_size(); ++j) {
            std::cout << "\tinput" << j << ": " << node.input(j) << std::endl;
        }
        for(int j = 0; j < node.output_size(); ++j) {
            std::cout << "\toutput" << j << ": " << node.output(j) << std::endl;
        }
        for(int j = 0; j < node.attribute_size(); ++j) {
            auto attribute = node.attribute(j);
            std::cout << "\tattribute" << j << ": " << attribute.name() << " ";
                      //<< "type: " << AttributeProto_AttributeType_Name(attribute.type()) << " ";
            if(attribute.has_f()) {
                std::cout << "float"
                          << ": " << attribute.f() << std::endl;
            }
            if(attribute.has_i()) {
                std::cout << "int"
                          << ": " << attribute.i() << std::endl;
            }
            if(attribute.has_s()) {
                std::cout << "string"
                          << ": " << attribute.s() << std::endl;
            }
            if(attribute.floats_size()) {
                std::cout << "floats: ";
                for(int k = 0; k < attribute.floats_size(); ++k) {
                    std::cout << attribute.floats(k) << " ";
                }
                std::cout << "\n";
            }
            if(attribute.ints_size()) {
                std::cout << "ints: ";
                for(int k = 0; k < attribute.ints_size(); ++k) {
                    std::cout << attribute.ints(k) << " ";
                }
                std::cout << "\n";
            }
            if(attribute.strings_size()) {
                std::cout << "strings: ";
                for(int k = 0; k < attribute.strings_size(); ++k) {
                    std::cout << attribute.strings(k) << " ";
                }
                std::cout << "\n";
            }
            /* TODO
            if (attribute.tensors_size()) {
                std::cout << "\t\ttensors";
            }
            */
        }
    }
}
