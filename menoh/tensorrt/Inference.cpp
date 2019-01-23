#include <chrono>
#include <iostream>

#include <menoh/array.hpp>
#include <menoh/exception.hpp>
#include <menoh/graph.hpp>
#include <menoh/utility.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <menoh/tensorrt/Exception.hpp>
#include <menoh/tensorrt/Inference.hpp>

using namespace nvinfer1;

#define CHECK(status)                             \
    do {                                          \
        auto ret = (status);                      \
        if(ret != 0) {                            \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while(0)

class Logger : public nvinfer1::ILogger {
public:
    Logger() : Logger(Severity::kWARNING) {}

    Logger(Severity severity) : reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override {
        // suppress messages with severity enum value greater than the
        // reportable
        if(severity > reportableSeverity)
            return;

        switch(severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity{Severity::kWARNING};
};

namespace menoh_impl {
    namespace tensorrt_backend {
#ifdef MENOH_ENABLE_TENSORRT_PROFILER
        struct Profiler : public IProfiler {
            const int TIMING_ITERATIONS = 1;

            typedef std::pair<std::string, float> Record;
            std::vector<Record> mProfile;

            virtual void reportLayerTime(const char* layerName, float ms) {
                auto record = std::find_if(
                  mProfile.begin(), mProfile.end(),
                  [&](const Record& r) { return r.first == layerName; });
                if(record == mProfile.end())
                    mProfile.push_back(std::make_pair(layerName, ms));
                else
                    record->second += ms;
            }

            void printLayerTimes() {
                float totalTime = 0;
                printf("\n=== Profiling ===\n");
                for(size_t i = 0; i < mProfile.size(); i++) {
                    printf("  %-40.40s %4.3f ms\n", mProfile[i].first.c_str(),
                           mProfile[i].second / TIMING_ITERATIONS);
                    totalTime += mProfile[i].second;
                }
                printf("=== Time over all layers: %4.3f ms ===\n\n",
                       totalTime / TIMING_ITERATIONS);
            }
        } gProfiler;
#endif
        static Logger gLogger;

        Inference::Inference(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data, config const& config)
          : config_(config) {

            std::vector<node> all_nodes;
            std::copy(model_data.node_list.begin(), model_data.node_list.end(),
                      back_inserter(all_nodes));

            for(auto const& name_and_arr_pair : input_table) {
                std::string name;
                array arr;
                std::tie(name, arr) = name_and_arr_pair;
                auto dims = arr.dims();

                input_name.push_back(name);

#ifdef MENOH_ENABLE_TENSORRT_DEBUG
                std::cout << "Input name(" << name << ") : dims("
                          << arr.dims().size() << ")  = ( ";
                for(auto size : arr.dims())
                    std::cerr << size << " ";
                std::cout << ")" << std::endl;
#endif
                m_Input[name] = arr;
                std::vector<std::string> inputs, outputs;
                outputs.push_back(name);
                std::unordered_map<std::string, attribute> attribute;
                attribute.insert({"dims", std::vector<int>(arr.dims().begin(),
                                                           arr.dims().end())});
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

                input_name.push_back(name);
#ifdef MENOH_ENABLE_TENSORRT_DEBUG
                std::cout << " Param : " << name << ", dims("
                          << arr.dims().size() << ")  = ( ";
                for(auto size : arr.dims())
                    std::cerr << size << " ";
                std::cout << ")" << std::endl;
#endif
                std::vector<std::string> inputs, outputs;
                outputs.push_back(name);
                std::unordered_map<std::string, attribute> attribute;
                attribute.insert({"dims", std::vector<int>(arr.dims().begin(),
                                                           arr.dims().end())});
                menoh_impl::node n{"Const", inputs, outputs, attribute};
                all_nodes.push_back(n);
            }

            if(output_table.size() == 0) {
                throw ParseException("output must have at least one entry");
            }

            for(auto const& name_and_arr : output_table) {
                std::string name;
                array arr;
                std::tie(name, arr) = name_and_arr;
                auto dims = arr.dims();
#ifdef MENOH_ENABLE_TENSORRT_DEBUG
                std::cout << "Output name(" << name << ") : dims("
                          << arr.dims().size() << ")  = ( ";
                for(auto size : arr.dims())
                    std::cerr << size << " ";
                std::cout << ")" << std::endl;
#endif
                m_Output[name] = arr;
            }

            {
                std::transform(output_table.begin(), output_table.end(),
                               std::back_inserter(output_name),
                               [](auto const& e) { return e.first; });
                std::sort(output_name.begin(), output_name.end());
            }

            if(output_name.size() == 0) {
                throw ParseException("outputs must have at least one entry");
            }

            auto graph = make_graph(all_nodes);

            Build(graph, parameter_table, output_name);
        }

        void Inference::Build(
          menoh_impl::graph& graph,
          std::unordered_map<std::string, array> const& parameter_table,
          std::vector<std::string>& outputs) {

            {
                int count;
                cudaGetDeviceCount(&count);
                if(count <= config_.device_id) {
                    throw ParseException("invalid device_id: " +
                                         std::to_string(config_.device_id) +
                                         " >= " + std::to_string(count) +
                                         " (available device count)");
                }
            }
            CHECK(cudaSetDevice(config_.device_id));
            builder.reset(createInferBuilder(gLogger));
            assert(builder);

            auto network = m_Parser.CreateNetwork(builder.get(), graph,
                                                  parameter_table, outputs);
            assert(network);

#ifdef MENOH_ENABLE_TENSORRT_DEBUG
            std::cout << "maxBatchSize = " << maxBatchSize << std::endl;
#endif
            builder->setMaxBatchSize(config_.max_batch_size);
            builder->setMaxWorkspaceSize(1 << 20);
            builder->setFp16Mode(false);
            builder->setDebugSync(false);

#ifdef MENOH_ENABLE_TENSORRT_PROFILER
            if(config_.enable_profiler) {
                std::cout << "buildCudaEngine::start" << std::endl;
                using clock = std::chrono::high_resolution_clock;
                auto start = clock::now();
#endif
                engine.reset(builder->buildCudaEngine(*network));
#ifdef MENOH_ENABLE_TENSORRT_PROFILER
                auto end = clock::now();
                std::cout
                  << "buildCudaEngine = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                       end - start)
                         .count() /
                       1000.0
                  << " sec" << std::endl;
                std::cout << "buildCudaEngine::done" << std::endl;
            } else {
                engine.reset(builder->buildCudaEngine(*network));
            }
#endif
            assert(engine);
            // we don't need the network any more
            network->destroy();
            context.reset(engine->createExecutionContext());
            assert(context);

#ifdef MENOH_ENABLE_TENSORRT_PROFILER
            if(config_.enable_profiler) {
                context->setProfiler(&gProfiler);
            }
#endif

            // allocate memory
            buffers_ = std::vector<void*>(engine->getNbBindings(), nullptr);
            for(auto const& p : m_Input) {
                auto name = p.first;
                auto index = engine->getBindingIndex(
                  m_Parser.ConvertToInputTensorName(name).c_str());
                input_memory_table_.emplace(p.first,
                                            make_cuda_memory_like(p.second));
                if(index == -1) {
                    throw ParseException("Input not found: " + name);
                }
                buffers_.at(index) = input_memory_table_.at(name).get();
            }
            for(auto const& p : m_Output) {
                auto name = p.first;
                auto index = engine->getBindingIndex(
                  m_Parser.ConvertToOutputTensorName(name).c_str());
                output_memory_table_.emplace(p.first,
                                             make_cuda_memory_like(p.second));
                if(index == -1) {
                    throw ParseException("Output not found: " + name);
                }
                buffers_.at(index) = output_memory_table_.at(name).get();
            }
        }

        // ==========================================================
        // Run
        // ==========================================================

        void Inference::Run() {

            auto runner = [&, this]() {
                cudaStream_t stream;
                CHECK(cudaStreamCreate(&stream));
                for(auto const& p : m_Input) {
                    auto const& name = p.first;
                    auto const& arr = p.second;
                    input_memory_table_.emplace(
                      p.first, make_cuda_memory_like(p.second));
                    CHECK(cudaMemcpyAsync(input_memory_table_.at(name).get(),
                                          static_cast<float*>(arr.data()),
                                          total_size_in_bytes(arr),
                                          cudaMemcpyHostToDevice, stream));
                }

                context->enqueue(config_.batch_size, buffers_.data(), stream,
                                 nullptr);

                for(auto const& p : m_Output) {
                    auto const& name = p.first;
                    auto const& arr = p.second;
                    output_memory_table_.emplace(
                      p.first, make_cuda_memory_like(p.second));
                    CHECK(cudaMemcpyAsync(static_cast<float*>(arr.data()),
                                          output_memory_table_.at(name).get(),
                                          total_size_in_bytes(arr),
                                          cudaMemcpyDeviceToHost, stream));
                }

                CHECK(cudaStreamSynchronize(stream));
                CHECK(cudaStreamDestroy(stream));
            };

#ifdef MENOH_ENABLE_TENSORRT_PROFILER
            if(config_.enable_profiler) {
                std::cout << "Inference::Run::start" << std::endl;
                using clock = std::chrono::high_resolution_clock;
                auto start = clock::now();
                runner();
                auto end = clock::now();
                std::cout
                  << "Run time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                       end - start)
                         .count() /
                       1000.0
                  << " sec" << std::endl;

                gProfiler.printLayerTimes();
                std::cout << "Inference::Run::done" << std::endl;
            } else {
                runner();
            }
#endif
        }

    } // namespace tensorrt_backend
} // namespace menoh_impl
