#ifndef MENOH_TENSORRT_CUDA_MEMORY_HPP
#define MENOH_TENSORRT_CUDA_MEMORY_HPP

#include <menoh/array.hpp>

namespace menoh_impl {
    namespace tensorrt_backend {

        class cuda_memory {
        public:
            cuda_memory(dtype_t dtype, std::vector<int> const& dims);

            void* get() const noexcept;

        private:
            dtype_t dtype_ = dtype_t::undefined;
            std::vector<int> dims_;

            struct cuda_memory_deleter {
                void operator()(void* p) const noexcept;
            };
            std::unique_ptr<void, cuda_memory_deleter> buffer_;
        };

        cuda_memory make_cuda_memory_like(array const& a);

    }
}

#endif // MENOH_TENSORRT_CUDA_MEMORY_HPP
