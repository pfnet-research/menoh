#ifndef TENSORRT_INFERENCE_HPP
#define TENSORRT_INFERENCE_HPP

#include <string>
#include <unordered_map>

#include <menoh/array.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>

#include <menoh/tensorrt/Parser.hpp>
#include <menoh/tensorrt/cuda_memory.hpp>

namespace menoh_impl {
    namespace tensorrt_backend {

        struct config {
            int batch_size;
            int max_batch_size;
            int device_id;
            bool enable_profiler;
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

            config config_;

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
