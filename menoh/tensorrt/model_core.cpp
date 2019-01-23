
#include <menoh/json.hpp>

#include <menoh/tensorrt/model_core.hpp>

namespace menoh_impl {
    namespace tensorrt_backend {

        model_core::model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data, config const& config)
          : m_inference(input_table, output_table, model_data, config) {}

        void model_core::do_run() { m_inference.Run(); }

        model_core make_model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data,
          backend_config const& config) {
            try {
                // default values
                int batch_size =
                  input_table.begin()->second.dims().front();
                int max_batch_size = batch_size;
                int device_id = 0;
                bool enable_profiler = false;
                int allow_fp16 = 0;
                int force_fp16 = 0;
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

                    if(c.find("device_id") != c.end()) {
                        device_id = c["device_id"].get<int>();
                    }
                    if(c.find("enable_profiler") != c.end()) {
                        enable_profiler = c["enable_profiler"].get<int>();
                    }
                    if(c.find("allow_fp16") != c.end()) {
                        allow_fp16 = c["allow_fp16"].get<int>();
                    }
                    if(c.find("force_fp16") != c.end()) {
                        force_fp16 = c["force_fp16"].get<int>();
                    }
                }
                assert(batch_size <= max_batch_size);
                tensorrt_backend::config config{batch_size, max_batch_size,
                                                device_id, enable_profiler};
                return model_core(input_table, output_table, model_data,
                                  config);
            } catch(nlohmann::json::parse_error const& e) {
                throw json_parse_error(e.what());
            }
        }

    } // namespace tensorrt_backend
} // namespace menoh_impl
