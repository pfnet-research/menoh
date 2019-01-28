#ifndef MENOH_TENSORRT_HOST_MEMORY
#define MENOH_TENSORRT_HOST_MEMORY

#include <NvInfer.h>

namespace menoh_impl {
    namespace tensorrt_backend {
        class host_memory {
        public:
            host_memory(nvinfer1::IHostMemory* m) : memory_(m) {}
            void* data() const { return memory_->data(); }
            std::size_t size() const { return memory_->size(); }
            nvinfer1::DataType type() const { return memory_->type(); }

        private:
            struct host_memory_deleter {
                void operator()(nvinfer1::IHostMemory* p) {
                    p->destroy();
                }
            };
            std::unique_ptr<nvinfer1::IHostMemory, host_memory_deleter> memory_;
        };

        inline void dump(host_memory const& m, std::ostream& os) {
            os.write(static_cast<const char*>(m.data()), m.size());
        }
    }
}

#endif // MENOH_TENSORRT_HOST_MEMORY
