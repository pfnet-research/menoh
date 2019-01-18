#include <cstddef>
#include <iostream>

#include <menoh/tensorrt/cuda_memory.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <menoh/array.hpp>
#include <menoh/tensorrt/Parser.hpp>

#define CHECK(status)                             \
    do {                                          \
        auto ret = (status);                      \
        if(ret != 0) {                            \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while(0)

namespace menoh_impl {
    namespace tensorrt_backend {

        cuda_memory::cuda_memory(dtype_t dtype, std::vector<int> const& dims)
          : dtype_(dtype), dims_(dims) {
            std::size_t size = 0;
            if(dtype_ == dtype_t::float_) {
                size = calc_total_size(dims) *
                       GetDataTypeSize(nvinfer1::DataType::kFLOAT);
            } else {
                throw std::runtime_error("unsupported dtype");
            }
            {
                void* raw_buff;
                CHECK(cudaMalloc(&raw_buff, size));
                // Be careful. Do NOT insert any codes in here
                buffer_.reset(raw_buff);
            }
        }

        void* cuda_memory::get() const noexcept { return buffer_.get(); }

        void cuda_memory::cuda_memory_deleter::operator()(void* p) const
          noexcept {
            CHECK(cudaFree(p));
        }

        cuda_memory make_cuda_memory_like(array const& a) {
            return cuda_memory(a.dtype(), a.dims());
        }
    }
}
