
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
          backend_config const& backend_config) {
            try {
                // default values
                int batch_size = input_table.begin()->second.dims().front();
                int max_batch_size = batch_size;
                int device_id = 0;
                bool enable_profiler = false;
                bool allow_fp16_mode = false;
                bool force_fp16_mode = false;
                bool enable_model_caching = true;
                std::string cached_model_dir = ".";
                if(!backend_config.empty()) {
                    auto c = nlohmann::json::parse(backend_config);
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
                        enable_profiler = c["enable_profiler"].get<bool>();
                    }
                    if(c.find("allow_fp16_mode") != c.end()) {
                        allow_fp16_mode = c["allow_fp16_mode"].get<bool>();
                    }
                    if(c.find("force_fp16_mode") != c.end()) {
                        force_fp16_mode = c["force_fp16_mode"].get<bool>();
                    }
                    if(c.find("enable_model_caching") != c.end()) {
                        enable_model_caching =
                          c["enable_model_caching"].get<bool>();
                    }
                    if(c.find("cached_model_dir") != c.end()) {
                        cached_model_dir =
                          c["cached_model_dir"].get<std::string>();
                    }
                }
                assert(batch_size <= max_batch_size);
                tensorrt_backend::config config{
                  backend_config,  batch_size,           max_batch_size,
                  device_id,       enable_profiler,      allow_fp16_mode,
                  force_fp16_mode, enable_model_caching, cached_model_dir};
                return model_core(input_table, output_table, model_data,
                                  config);
            } catch(nlohmann::json::parse_error const& e) {
                throw json_parse_error(e.what());
            }
        }

    } // namespace tensorrt_backend
} // namespace menoh_impl
