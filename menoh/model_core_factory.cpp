#include <menoh/model_core.hpp>
#include <menoh/model_core_factory.hpp>

<<<<<<< HEAD
#if ENABLE_MKLDNN
#include <menoh/mkldnn/model_core.hpp>
#endif
#if ENABLE_ARMNN
#include <menoh/armnn/model_core.hpp>
#endif

#include <menoh/composite_backend/model_core.hpp>

#include <menoh/json.hpp>

namespace menoh_impl {

#if ENABLE_MKLDNN
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
        }
#endif

#if ENABLE_ARMNN
        if(backend_name == "armnn") {
            return std::make_unique<armnn_backend::model_core>(
              armnn_backend::make_model_core(input_table, required_output_table,
                                              model_data, config));
        }
#endif

        throw invalid_backend_name(backend_name);
    }

} // namespace menoh_impl
