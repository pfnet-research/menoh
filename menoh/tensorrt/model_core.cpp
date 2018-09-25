
#include <menoh/json.hpp>

#include <menoh/tensorrt/model_core.hpp>

namespace menoh_impl {
    namespace tensorrt_backend {

        model_core::model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data )
          : m_inference(Params(&input_table, &output_table, &model_data)) {}

        void model_core::do_run() {
            m_inference.Run();
        }

        model_core make_model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data,
          backend_config const& config) {
            try {
                return model_core(input_table, output_table, model_data);
            } catch(nlohmann::json::parse_error const& e) {
                throw json_parse_error(e.what());
            }
        }

    } // namespace tensorrt_backend
} // namespace menoh_impl
