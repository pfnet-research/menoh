#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <menoh/array.hpp>
#include <menoh/exception.hpp>
#include <menoh/graph.hpp>
#include <menoh/utility.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <menoh/hash/hasher.hpp>

#include <menoh/tensorrt/Exception.hpp>
#include <menoh/tensorrt/Inference.hpp>
#include <menoh/tensorrt/host_memory.hpp>

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
#endif // MENOH_ENABLE_TENSORRT_PROFILER
        static Logger gLogger;

#ifdef MENOH_ENABLE_TENSORRT_MODEL_CACHING
        std::string Inference::calc_model_hash(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data, config const& config) {
            hasher hasher;
            auto add_variable_table = [](
              menoh_impl::hasher& h,
              std::unordered_map<std::string, array> const& table) {
                std::vector<std::pair<std::string, array>> variables(
                  table.begin(), table.end());
                std::sort(variables.begin(), variables.end(),
                          [](auto const& a, auto const& b) {
                              return a.first < b.first;
                          });
                for(auto const& p : variables) {
                    add_container(h, p.first);
                    h.add(
                      static_cast<const std::uint8_t*>(
                        static_cast<const void*>(p.second.dims().data())),
                      p.second.dims().size() *
                        sizeof(
                          std::decay_t<decltype(p.second.dims())>::value_type));
                    // Do not process values in p.second because it is a
                    // placeholder
                }
            };
            add_variable_table(hasher, input_table);  // [input_table]
            add_variable_table(hasher, output_table); // [output_table]
            auto add_str_vec = [](menoh_impl::hasher& h,
                                  std::vector<std::string> const& sv) {
                std::for_each(sv.begin(), sv.end(),
                              [&h](auto const& s) { add_container(h, s); });
            };
            auto add_attr = [](menoh_impl::hasher& h, auto const& a) {
                auto* i = get_if<int>(&a);
                auto* f = get_if<float>(&a);
                auto* is = get_if<std::vector<int>>(&a);
                auto* fs = get_if<std::vector<float>>(&a);
                if(i) {
                    add_container(h, "int" + std::to_string(*i));
                } else if(f) {
                    add_container(h, "float" + std::to_string(*f));
                } else if(is) {
                    add_c_str(h, "ints");
                    std::for_each(is->begin(), is->end(), [&h](auto i) {
                        add_container(h, std::to_string(i));
                    });
                } else if(fs) {
                    add_c_str(h, "floats");
                    std::for_each(fs->begin(), fs->end(), [&h](auto f) {
                        add_container(h, std::to_string(f));
                    });
                } else {
                    // TODO other types
                    assert(!"Don't come here");
                }
            };

            // [model_data]
            for(auto const& node : model_data.node_list) {
                add_container(hasher, node.op_type);
                add_str_vec(hasher, node.input_name_list);
                add_str_vec(hasher, node.output_name_list);
                std::vector<std::pair<std::string, attribute>> attributes(
                  node.attribute_table.begin(), node.attribute_table.end());
                std::sort(attributes.begin(), attributes.end(),
                          [](auto const& a, auto const& b) {
                              return a.first < b.first;
                          });
                std::for_each(attributes.begin(), attributes.end(),
                              [&hasher, add_attr](auto const& p) {
                                  add_container(hasher, p.first);
                                  add_attr(hasher, p.second);
                              });
            }
            for(auto const& p : model_data.parameter_name_and_array_list) {
                add_container(hasher, p.first);
                add_container(hasher, p.second.dims());
                hasher.add(static_cast<std::uint8_t*>(p.second.data()),
                           total_size_in_bytes(p.second));
            }
            if(config.config_json_object_opt) {
                auto j = *config.config_json_object_opt;
                // ignore options independent from model itself
                j.erase("device_id");
                j.erase("enable_model_caching");
                j.erase("cached_model_dir");
                if(!j.empty()) {
                    add_container(hasher, j.dump()); // [config]
                }
            }

            cudaDeviceProp device_prop;
            cudaGetDeviceProperties(&device_prop, config_.device_id);
            add_container(hasher, std::string(device_prop.name));
            return hasher.finish();
        }
#endif // MENOH_ENABLE_TENSORRT_MODEL_CACHING

        Inference::Inference(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data, config const& config)
          : config_(config) {
#ifdef MENOH_ENABLE_TENSORRT_MODEL_CACHING
            if(config_.enable_model_caching) {
#ifdef MENOH_ENABLE_TENSORRT_PROFILER
                if(config_.enable_profiler) {
                    std::cout << "calc_model_hash::start" << std::endl;
                    using clock = std::chrono::high_resolution_clock;
                    auto start = clock::now();

                    model_hash_ = calc_model_hash(input_table, output_table,
                                                  model_data, config);

                    auto end = clock::now();
                    std::cout
                      << "calc_model_hash = "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                           end - start)
                             .count() /
                           1000.0
                      << " sec" << std::endl;
                    std::cout << "calc_model_hash::done" << std::endl;
                } else {
#endif // MENOH_ENABLE_TENSORRT_PROFILER
                    model_hash_ = calc_model_hash(input_table, output_table,
                                                  model_data, config);
#ifdef MENOH_ENABLE_TENSORRT_PROFILER
                }
#endif // MENOH_ENABLE_TENSORRT_PROFILER
#ifdef MENOH_ENABLE_TENSORRT_DEBUG
                std::cout << "model_hash: " << model_hash_ << std::endl;
#endif // MENOH_ENABLE_TENSORRT_DEBUG
            }
#endif // MENOH_ENABLE_TENSORRT_MODEL_CACHING
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
#endif // MENOH_ENABLE_TENSORRT_DEBUG
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
#endif // MENOH_ENABLE_TENSORRT_DEBUG
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
#endif // MENOH_ENABLE_TENSORRT_DEBUG
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

            { // device_id check
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

            auto build_cuda_engine = [&, this]() {
                builder.reset(createInferBuilder(gLogger));
                assert(builder);

                unique_ptr_with_destroyer<INetworkDefinition> network(
                  m_Parser.CreateNetwork(builder.get(), graph, parameter_table,
                                         outputs));
                assert(network);

#ifdef MENOH_ENABLE_TENSORRT_DEBUG
                std::cout << "maxBatchSize = " << maxBatchSize << std::endl;
#endif // MENOH_ENABLE_TENSORRT_DEBUG
                builder->setMaxBatchSize(config_.max_batch_size);
                builder->setMaxWorkspaceSize(1 << 20);
                if(config_.force_fp16_mode) {
                    if(!builder->platformHasFastFp16()) {
                        throw ParseException(
                          "FP16 mode is not available on this device");
                    }
                    builder->setFp16Mode(true);
                    builder->setStrictTypeConstraints(true);
                } else if(config_.allow_fp16_mode &&
                          builder->platformHasFastFp16()) {
                    builder->setFp16Mode(true);
                }
                builder->setDebugSync(false);

#ifdef MENOH_ENABLE_TENSORRT_PROFILER
                if(config_.enable_profiler) {
                    std::cout << "buildCudaEngine::start" << std::endl;
                    using clock = std::chrono::high_resolution_clock;
                    auto start = clock::now();

                    engine.reset(builder->buildCudaEngine(*network));

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
#endif // MENOH_ENABLE_TENSORRT_PROFILER
                    engine.reset(builder->buildCudaEngine(*network));
#ifdef MENOH_ENABLE_TENSORRT_PROFILER
                }
#endif // MENOH_ENABLE_TENSORRT_PROFILER
            };

// build or deserialize engine
#ifdef MENOH_ENABLE_TENSORRT_MODEL_CACHING
            if(config_.enable_model_caching) {
                auto model_cache_path =
                  config_.cached_model_dir + "/" + model_hash_ + ".trt";
                std::ifstream ifs(model_cache_path, std::ios::binary);
                if(ifs.good()) { // exist cache
                    // load cache
                    auto load_cached_engine = [&, this]() {
                        auto begin = ifs.tellg();
                        ifs.seekg(0, std::ios::end);
                        auto file_size = ifs.tellg() - begin;
                        std::vector<std::uint8_t> data(file_size);
                        ifs.seekg(0, std::ios_base::beg);
                        ifs.read(
                          static_cast<char*>(static_cast<void*>(data.data())),
                          file_size);
                        ifs.close();
                        runtime.reset(nvinfer1::createInferRuntime(gLogger));
                        // deserialize engine
                        engine.reset(runtime->deserializeCudaEngine(
                          data.data(), data.size(), nullptr));
                    };
#ifdef MENOH_ENABLE_TENSORRT_PROFILER
                    if(config_.enable_profiler) {
                        std::cout << "Inference::LoadCachedEngine::start"
                                  << std::endl;
                        using clock = std::chrono::high_resolution_clock;
                        auto start = clock::now();

                        load_cached_engine();

                        auto end = clock::now();
                        std::cout << "LoadCachedEngine = "
                                  << std::chrono::duration_cast<
                                       std::chrono::milliseconds>(end - start)
                                         .count() /
                                       1000.0
                                  << " sec" << std::endl;
                        std::cout << "Inference::LoadCachedEngine::done"
                                  << std::endl;

                        gProfiler.printLayerTimes();
                    } else {
#endif // MENOH_ENABLE_TENSORRT_PROFILER
                        load_cached_engine();
#ifdef MENOH_ENABLE_TENSORRT_PROFILER
                    }
#endif // MENOH_ENABLE_TENSORRT_PROFILER
                } else {
                    // not exist cache
                    build_cuda_engine();

                    // make cache
                    std::ofstream ofs(model_cache_path, std::ios::binary);
                    host_memory serialized_engine(engine->serialize());
                    dump(serialized_engine, ofs);
                }
            } else {
#endif // MENOH_ENABLE_TENSORRT_MODEL_CACHING
                build_cuda_engine();
#ifdef MENOH_ENABLE_TENSORRT_MODEL_CACHING
            }
#endif // MENOH_ENABLE_TENSORRT_MODEL_CACHING
            assert(engine);

            context.reset(engine->createExecutionContext());
            assert(context);

#ifdef MENOH_ENABLE_TENSORRT_PROFILER
            if(config_.enable_profiler) {
                context->setProfiler(&gProfiler);
            }
#endif // MENOH_ENABLE_TENSORRT_PROFILER

            // allocate memory
            buffers_ = std::vector<void*>(engine->getNbBindings(), nullptr);
            for(auto const& p : m_Input) {
                auto name = p.first;
                auto index = engine->getBindingIndex(name.c_str());
                input_memory_table_.emplace(p.first,
                                            make_cuda_memory_like(p.second));
                if(index == -1) {
                    throw ParseException("Input not found: " + name);
                }
                buffers_.at(index) = input_memory_table_.at(name).get();
            }
            for(auto const& p : m_Output) {
                auto name = p.first;
                auto index = engine->getBindingIndex(name.c_str());
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
                std::cout << "Inference::Run::done" << std::endl;
                std::cout
                  << "Run time = "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                       end - start)
                       .count()
                  << " msec" << std::endl;

                gProfiler.printLayerTimes();
            } else {
#endif // MENOH_ENABLE_TENSORRT_PROFILER
                runner();
#ifdef MENOH_ENABLE_TENSORRT_PROFILER
            }
#endif // MENOH_ENABLE_TENSORRT_PROFILER
        }

    } // namespace tensorrt_backend
} // namespace menoh_impl
