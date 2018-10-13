#ifndef MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_MKLDNN_MEMORY_CACHE_HPP
#define MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_MKLDNN_MEMORY_CACHE_HPP

#include <numeric>
#include <tuple>

#include <menoh/array.hpp>
#include <menoh/optional.hpp>

#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_conversion.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            class memory_cache {
            public:
                memory_cache() = default;

                memory_cache(array const& arr, mkldnn::engine const& engine)
                  : original_array_(arr), engine_(engine) {}
                memory_cache(mkldnn::memory const& mem)
                  : cached_memory_list_({mem}),
                    engine_(mem.get_primitive_desc().get_engine()) {}

                mkldnn::memory::data_type dtype() const {
                    if(original_array_) {
                        return dtype_to_mkldnn_memory_data_type(
                          original_array_->dtype());
                    }
                    assert(!cached_memory_list_.empty());
                    return extract_data_type(cached_memory_list_.front());
                }

                std::vector<int> dims() const {
                    if(original_array_) {
                        return original_array_->dims();
                    }
                    assert(!cached_memory_list_.empty());
                    return extract_dims(cached_memory_list_.front());
                }

                mkldnn::engine engine() const { return *engine_; }

                std::tuple<mkldnn::memory, optional<mkldnn::primitive>>
                get_memory(std::vector<int> const& dims,
                           mkldnn::memory::format format);

                mkldnn::memory get_data_memory();

                void add_cached_memory(mkldnn::memory const& added_memory) {
                    // check format is different
                    // MEMO: dims may be different (eg FC's weight for 4d input
                    // and 2d input)
                    if(original_array_) {
                        auto mdims = extract_dims(added_memory);
                        assert(total_size(*original_array_) ==
                               std::accumulate(mdims.begin(), mdims.end(), 1,
                                               std::multiplies<int>()));
                    }
                    for(auto const& cached_memory : cached_memory_list_) {
                        assert(extract_format(cached_memory) !=
                               extract_format(added_memory));
                    }

                    cached_memory_list_.push_back(added_memory);
                }

            private:
                optional<array> original_array_ = nullopt;
                std::vector<mkldnn::memory> cached_memory_list_;
                optional<mkldnn::engine> engine_;
            };

            mkldnn::memory inline get_memory(
              memory_cache& mem_cache, std::vector<int> const& dims,
              mkldnn::memory::format format,
              std::vector<mkldnn::primitive>& primitives) {
                optional<mkldnn::memory> mem;
                optional<mkldnn::primitive> reorder_primitive;
                std::tie(mem, reorder_primitive) =
                  mem_cache.get_memory(dims, format);
                if(reorder_primitive) {
                    primitives.push_back(*reorder_primitive);
                }
                return *mem;
            }

            mkldnn::memory inline get_memory(
              memory_cache& mem_cache, mkldnn::memory::format format,
              std::vector<mkldnn::primitive>& primitives) {
                return get_memory(mem_cache, mem_cache.dims(), format,
                                  primitives);
            }

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_MKLDNN_WITH_GENERIC_FALLBACK_BACKEND_MKLDNN_MEMORY_CACHE_HPP
