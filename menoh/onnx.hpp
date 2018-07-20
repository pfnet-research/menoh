#ifndef MENOH_ONNX_HPP
#define MENOH_ONNX_HPP

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <menoh/exception.hpp>

namespace menoh_impl {

    struct model_data;

    class onnx_parse_error : public exception {
    public:
        onnx_parse_error(std::string const& filename)
          : exception(menoh_error_code_onnx_parse_error,
                      "menoh onnx parse error: " + filename) {}
    };

    class unsupported_onnx_opset_version : public exception {
    public:
        unsupported_onnx_opset_version(std::string const& filename,
                                       int actual_version,
                                       int supported_version)
          : exception(
              menoh_error_code_unsupported_onnx_opset_version,
              "menoh unsupported onnx opset version error: " + filename +
                " has onnx opset version " + std::to_string(actual_version) +
                " > " + std::to_string(supported_version)) {}
    };

    class invalid_attribute_type : public exception {
    public:
        invalid_attribute_type(std::string const& attribute_name,
                               std::string const& attribute_type)
          : exception(menoh_error_code_invalid_attribute_type,
                      "menoh invalid attribute type error: attribute type " +
                        attribute_type + " for \"" + attribute_name + "\"") {}
    };

    model_data load_onnx(std::string const& filename);

    std::vector<std::string>
    extract_model_input_name_list(menoh_impl::model_data const& model_data);

    std::vector<std::string>
    extract_model_output_name_list(menoh_impl::model_data const& model_data);

} // namespace menoh_impl

#endif // MENOH_ONNX_HPP
