
#include <menoh/json.hpp>

#include <menoh/tensorrt/model_core.hpp>

namespace menoh_impl {
    namespace tensorrt_backend {

        model_core::model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data,
          int batch_size, int max_batch_size)
          : m_inference(Params(&input_table, &output_table, &model_data,
                               batch_size, max_batch_size)) {}

        void model_core::do_run() {
            m_inference.Run();
        }

        model_core make_model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data,
          backend_config const& config) {
            try {
                int batch_size = input_table.begin()->second.dims().front(); // default
                int max_batch_size = batch_size; // default
                if(!config.empty()) {
                    auto c = nlohmann::json::parse(config);
                    if(c.find("batch_size") != c.end()) {
                        batch_size = c["batch_size"].get<int>();
                    }
                    if(c.find("max_batch_size") != c.end()) {
                        max_batch_size = c["max_batch_size"].get<int>();
                    }
                    if(max_batch_size < batch_size) {
                        max_batch_size = batch_size;
                    }
                }
                assert(batch_size <= max_batch_size);
                return model_core(input_table, output_table, model_data,
                                  batch_size, max_batch_size);
            } catch(nlohmann::json::parse_error const& e) {
                throw json_parse_error(e.what());
            }
        }

    } // namespace tensorrt_backend
} // namespace menoh_impl
