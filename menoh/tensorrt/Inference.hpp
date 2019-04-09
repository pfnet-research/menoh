#ifndef TENSORRT_INFERENCE_HPP
#define TENSORRT_INFERENCE_HPP

#include <string>
#include <unordered_map>

#include <menoh/array.hpp>
#include <menoh/json.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>
#include <menoh/optional.hpp>

#include <menoh/tensorrt/Parser.hpp>
#include <menoh/tensorrt/cuda_memory.hpp>

namespace menoh_impl {
    namespace tensorrt_backend {

        struct config {
            optional<nlohmann::json> config_json_object_opt; // for hashing
            int batch_size;
            int max_batch_size;
            int device_id;
            bool enable_profiler;
            bool allow_fp16_mode;
            bool force_fp16_mode;
            bool enable_model_caching;
            std::string cached_model_dir;
        };

        class Inference {
        public:
            Inference(
              std::unordered_map<std::string, array> const& input_table,
              std::unordered_map<std::string, array> const& output_table,
              menoh_impl::model_data const& model_data, config const& config);

            void Run();

        private:

            void
            Build(graph& menoh_graph,
                  std::unordered_map<std::string, array> const& parameter_table,
                  std::vector<std::string>& outputs);

#ifdef MENOH_ENABLE_TENSORRT_PROFILER
            class profiler : public nvinfer1::IProfiler {
            public:
                virtual void reportLayerTime(const char* layer_name,
                                             float ms) override {
                    auto record = profile_.find(layer_name);
                    if(record == profile_.end()) {
                        profile_.emplace(layer_name, ms);
                    } else {
                        record->second += ms;
                    }
                }

                void print_layer_times() const {
                    std::vector<std::pair<std::string, float>> record_list(
                      profile_.begin(), profile_.end());
                    std::sort(record_list.begin(), record_list.end(),
                              [](auto const& a, auto const& b) {
                                  return a.second > b.second;
                              });

                    float total_time = 0;
                    std::printf("\n=== Profiling ===\n");
                    for(auto const& r : record_list) {
                        std::printf("  %-40.40s %4.3f ms\n", r.first.c_str(),
                                    r.second / timing_iterations);
                        total_time += r.second;
                    }
                    std::printf("=== Time over all layers: %4.3f ms ===\n\n",
                                total_time / timing_iterations);
                }

            private:
                static constexpr int timing_iterations = 1;

                std::unordered_map<std::string, float> profile_;
            };
            std::unique_ptr<profiler> profiler_{std::make_unique<
              profiler>()}; // FIXME this indirection seems unnecessary

#endif // MENOH_ENABLE_TENSORRT_PROFILER

            config config_;
            std::string model_hash_;

            Parser m_Parser;

            template <typename T>
            struct destroyer {
                void operator()(T* p) const noexcept { p->destroy(); }
            };
            template <typename T>
            using unique_ptr_with_destroyer = std::unique_ptr<T, destroyer<T>>;

            unique_ptr_with_destroyer<IBuilder> builder;
            unique_ptr_with_destroyer<ICudaEngine> engine;
            unique_ptr_with_destroyer<IExecutionContext> context;

#ifdef MENOH_ENABLE_TENSORRT_MODEL_CACHING
            std::string calc_model_hash(
              std::unordered_map<std::string, array> const& input_table,
              std::unordered_map<std::string, array> const& output_table,
              menoh_impl::model_data const& model_data, config const& config);

            unique_ptr_with_destroyer<IRuntime> runtime;
#endif // MENOH_ENABLE_TENSORRT_MODEL_CACHING

            std::vector<std::string> input_name;
            std::vector<std::string> output_name;

            std::unordered_map<std::string, array> m_Input;
            std::unordered_map<std::string, array> m_Output;

            std::unordered_map<std::string, cuda_memory> input_memory_table_;
            std::unordered_map<std::string, cuda_memory> output_memory_table_;
            std::vector<void*> buffers_;
        };

    } // namespace tensorrt_backend
} // namespace menoh_impl
#endif // TENSORRT_INFERENCE_HPP
