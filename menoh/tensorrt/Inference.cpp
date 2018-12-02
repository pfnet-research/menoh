#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <tuple>

#include <menoh/array.hpp>
#include <menoh/exception.hpp>
#include <menoh/graph.hpp>
#include <menoh/json.hpp>
#include <menoh/utility.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <menoh/tensorrt/Inference.hpp>

using namespace nvinfer1;

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

class Logger : public nvinfer1::ILogger
{
public:
  
    Logger(): Logger(Severity::kWARNING) {}

    Logger(Severity severity): reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity{Severity::kWARNING};
};

namespace menoh_impl {
    namespace tensorrt_backend {

        struct Profiler : public IProfiler
        {
            const int TIMING_ITERATIONS = 1;

            typedef std::pair<std::string, float> Record;
            std::vector<Record> mProfile;

            virtual void reportLayerTime(const char* layerName, float ms)
            {
                auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
                if (record == mProfile.end())
                    mProfile.push_back(std::make_pair(layerName, ms));
                else
                    record->second += ms;
            }

            void printLayerTimes()
            {
                float totalTime = 0;
                printf("\n=== Profiling ===\n");
                for (size_t i = 0; i < mProfile.size(); i++)
                {
                    printf("  %-40.40s %4.3f ms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
                    totalTime += mProfile[i].second;
                }
                printf("=== Time over all layers: %4.3f ms ===\n\n", totalTime / TIMING_ITERATIONS);
            }
        } gProfiler;
      
        static Logger gLogger;

        Inference::Inference( const Params& params )
          : m_Parser()
          , batchSize(params.batchSize)
          , maxBatchSize(params.maxBatchSize)
          , m_Network(nullptr)
          , builder(nullptr)
          , engine(nullptr)
          , context(nullptr)
        {
            menoh_impl::model_data const& model_data = *(params.model_data_);

            std::vector<node> all_nodes;
            std::copy ( model_data.node_list.begin(), model_data.node_list.end(), back_inserter(all_nodes) );

            std::map<std::string, TensorShape> inputShapes;

            std::unordered_map<std::string, array> const& input_table = *(params.input_table_);
            for(auto const& name_and_arr_pair : input_table) {
                std::string name;
                array arr;
                std::tie(name, arr) = name_and_arr_pair;
                auto dims = arr.dims();
                inputShapes[name] = TensorShape(dims.size(), (const unsigned int*)dims.data());
                input_name_list.push_back(name);
#ifdef TENSORRT_DEBUG  
                std::cout << "Input name(" << name << ") : dims(" << arr.dims().size() << ")  = ( ";
                for( auto size : arr.dims() ) std::cerr << size << " "; 
                std::cout << ")" << std::endl;
#endif
                m_Input[name] = arr;
                std::vector<std::string> inputs, outputs;
                outputs.push_back(name);
                std::unordered_map<std::string, attribute> attribute;
                attribute.insert({"dims", std::vector<int>(arr.dims().begin(), arr.dims().end())});
                menoh_impl::node n{"Placeholder", inputs, outputs, attribute};
                all_nodes.push_back(n);
            }            

            auto parameter_table = std::unordered_map<std::string, array>(
                                        model_data.parameter_name_and_array_list.begin(),
                                        model_data.parameter_name_and_array_list.end());
            for(auto const& param : parameter_table) {
                std::string name;
                array arr;
                std::tie(name, arr) = param;
                auto dims = arr.dims();
                inputShapes[name] = TensorShape(dims.size(), (const unsigned int*)dims.data());
                input_name_list.push_back(name);
#ifdef TENSORRT_DEBUG
                std::cout << " Param : " << name << ", dims(" << arr.dims().size() << ")  = ( ";
                for( auto size : arr.dims() ) std::cerr << size << " "; 
                std::cout << ")" << std::endl;
#endif                
                std::vector<std::string> inputs, outputs;
                outputs.push_back(name);
                std::unordered_map<std::string, attribute> attribute;
                attribute.insert({"dims", std::vector<int>(arr.dims().begin(), arr.dims().end())});
                menoh_impl::node n{"Const", inputs, outputs, attribute};
                all_nodes.push_back(n);
            }            

            std::unordered_map<std::string, array> const& output_table = *(params.output_table_);
            if (output_table.size() == 0)
            {
                throw ParseException("output must have at least one entry");
            }
 
            for(auto const& name_and_arr_pair : output_table) {
                std::string name;
                array arr;
                std::tie(name, arr) = name_and_arr_pair;
                auto dims = arr.dims();
#ifdef TENSORRT_DEBUG
                std::cout << "Output name(" << name << ") : dims(" << arr.dims().size() << ")  = ( ";
                for( auto size : arr.dims() ) std::cerr << size << " "; 
                std::cout << ")" << std::endl;
#endif
                m_Output[name] = arr;
            }            

            {
                std::transform(output_table.begin(), output_table.end(),
                               std::back_inserter(output_name_sorted_list),
                               [](auto const& e) { return e.first; });
                std::sort(output_name_sorted_list.begin(),
                          output_name_sorted_list.end());
            }
            std::vector<std::string> requestedOutputs{ output_name_sorted_list };

            auto graph = make_graph(all_nodes);            

            Build( graph, parameter_table, inputShapes, requestedOutputs );
        }

        // ==========================================================
        // Build
        // ==========================================================

        void Inference::Build( menoh_impl::graph& graph,
                                       std::unordered_map<std::string, array> const& parameter_table,
                                       std::map<std::string, TensorShape>& inputShapes,
                                       std::vector<std::string>& requestedOutputs ) {

            builder = createInferBuilder(gLogger);
            assert(builder); 

            m_Network = m_Parser.CreateNetworkFromGraph( builder,
                                                         graph,
                                                         parameter_table,
                                                         inputShapes,
                                                         requestedOutputs );
            assert(m_Network); 

#ifdef TENSORRT_DEBUG
            std::cout << "maxBatchSize = " << maxBatchSize << std::endl;
#endif
            builder->setMaxBatchSize(maxBatchSize);
            builder->setMaxWorkspaceSize(1 << 20);

            builder->setFp16Mode(false);

            builder->setDebugSync(false);

            {
                std::cout << "buildCudaEngine::start" << std::endl;
                using clock = std::chrono::high_resolution_clock;
                auto start = clock::now();
                engine = builder->buildCudaEngine(*m_Network);
                auto end = clock::now();
                std::cout << "buildCudaEngine = "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(end -start).count()/1000.0
                          << " sec" << std::endl;
                std::cout << "buildCudaEngine::done" << std::endl;
                assert(engine);
            }
        }
  
        // ==========================================================
        // Run
        // ==========================================================

        void Inference::Run() {
#ifdef TENSORRT_DEBUG
            std::cout << "Inference::Run::start" << std::endl;
#endif
            using clock = std::chrono::high_resolution_clock;
            auto start = clock::now();

            context = engine->createExecutionContext();
            assert(context);

            context->setProfiler(&gProfiler);

            auto input_map  = m_Input[ input_name_list[0].c_str()];
            auto output_map = m_Output[output_name_sorted_list[0].c_str()];
            int input_size  = total_size(input_map) *GetDataTypeSize(DataType::kFLOAT);
            int output_size = total_size(output_map)*GetDataTypeSize(DataType::kFLOAT);

#ifdef TENSORRT_DEBUG
            std::cout << "Run, input_size = " << input_size << ", output_size = " << output_size << std::endl;            
#endif
            {
                void* buffers[2];
                AllocateMemory(&buffers[0], input_size  );
                AllocateMemory(&buffers[1], output_size );

                cudaStream_t stream;
                CHECK(cudaStreamCreate(&stream));

                PushMemory(buffers[0], (float*)input_map.data(),  input_size,  stream);

                context->enqueue(batchSize, buffers, stream, nullptr);

                PullMemory(buffers[1], (float*)output_map.data(), output_size, stream);

                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);

                FreeMemory(buffers[0]);
                FreeMemory(buffers[1]);
            }

            context->destroy();
            engine->destroy();
            builder->destroy();

            auto end = clock::now();
            std::cout << "Run time = "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end -start).count()/1000.0
                      << " sec" << std::endl;

            gProfiler.printLayerTimes();
            
#ifdef TENSORRT_DEBUG
            std::cout << "Inference::Run::done" << std::endl;          
#endif
        }

        // ==========================================================
        // support methods
        // ==========================================================

        void Inference::AllocateMemory(void** buffer, int size) {
            CHECK(cudaMalloc(buffer, size));
        }

        void Inference::PushMemory(void* buffer, float* input, int size, cudaStream_t stream) {
            CHECK(cudaMemcpyAsync(buffer, input, size, cudaMemcpyHostToDevice, stream));
        }
      
        void Inference::PullMemory(void* buffer, float* output, int size, cudaStream_t stream) {
            CHECK(cudaMemcpyAsync(output, buffer, size, cudaMemcpyDeviceToHost, stream));
        }
      
        void Inference::FreeMemory(void* buffer) {
            CHECK(cudaFree(buffer));
        }
  
    } // namespace tensorrt_backend
} // namespace menoh_impl
