#include <menoh/model_core.hpp>
#include <menoh/model_core_factory.hpp>

#include <menoh/composite_backend/model_core.hpp>
#include <menoh/mkldnn/model_core.hpp>
#include <menoh/onnxifi/model_core.hpp>

#include <menoh/json.hpp>

namespace menoh_impl {

    std::unique_ptr<menoh_impl::model_core> make_model_core(
      std::unordered_map<std::string, array> const& input_table,
      std::unordered_map<std::string, array> const& required_output_table,
      std::unordered_map<std::string, array_profile> const&
        output_profile_table,
      menoh_impl::model_data const& model_data, std::string const& backend_name,
      backend_config const& config) {
        if(backend_name == "mkldnn") {
            return std::make_unique<mkldnn_backend::model_core>(
              mkldnn_backend::make_model_core(
                input_table, required_output_table, model_data, config));
        } else if(backend_name == "mkldnn_with_generic_fallback") {
            auto conf = nlohmann::json::parse(config.empty() ? "{}" : config);
            conf.merge_patch(
              R"({"backends":[{"type":"mkldnn"}, {"type":"generic"}]})");
            return std::make_unique<composite_backend::model_core>(
              composite_backend::make_model_core(
                input_table, required_output_table, output_profile_table,
                model_data, conf.get<std::string>()));
        } else if(backend_name == "composite_backend") {
            return std::make_unique<composite_backend::model_core>(
              composite_backend::make_model_core(
                input_table, required_output_table, output_profile_table,
                model_data, config));
        } else if(backend_name == "onnxifi") {
            return std::make_unique<onnxifi_backend::model_core>(
              onnxifi_backend::make_model_core(
                input_table, required_output_table, output_profile_table,
                model_data, config));
        }

        throw invalid_backend_name(backend_name);
    }

} // namespace menoh_impl
