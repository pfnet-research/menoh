#ifndef TENSORRT_INFERENCE_HPP
#define TENSORRT_INFERENCE_HPP

#include <menoh/array.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>

#include <menoh/tensorrt/Parser.hpp>

namespace menoh_impl {
    namespace tensorrt_backend {

        struct Params
        {
            Params(
                std::unordered_map<std::string, array> const* input_table,
                std::unordered_map<std::string, array> const* output_table,
                menoh_impl::model_data const* model_data,
                int batch_size, int max_batch_size, int device_id)
	        : batchSize(batch_size)
	        , maxBatchSize(max_batch_size)
            , device_id(device_id)
	        , input_table_(input_table)
                , output_table_(output_table)
	        , model_data_(model_data) {}

            int batchSize;
            int maxBatchSize; 
            int device_id;
            std::unordered_map<std::string, array> const* input_table_;
            std::unordered_map<std::string, array> const* output_table_;
            menoh_impl::model_data const* model_data_;
        };

        class Inference {
        public:

            Inference(const Params& param);

            void Run();
            void Clear();

        private:

            void Build(graph& menoh_graph,
                       std::unordered_map<std::string, array> const& parameter_table,
                       std::vector<std::string>& outputs);

            Parser              m_Parser;
            int                 batchSize;
            int                 maxBatchSize;
            int                 device_id;

            INetworkDefinition  *m_Network;
            IBuilder            *builder;
            ICudaEngine         *engine;
            IExecutionContext   *context;

            std::vector<std::string> input_name;
            std::vector<std::string> output_name;

            std::unordered_map<std::string, array> m_Input;
            std::unordered_map<std::string, array> m_Output;
        };

    } // namespace tensorrt_backend
} // namespace menoh_impl
#endif // TENSORRT_INFERENCE_HPP
