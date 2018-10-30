#include <menoh/mkldnn_with_generic_fallback/backend/mkldnn/memory_cache.hpp>

#include <numeric> // for accumulate

namespace menoh_impl {
    namespace mkldnn_with_generic_fallback_backend {
        namespace mkldnn_backend {

            std::tuple<mkldnn::memory, optional<mkldnn::primitive>>
            memory_cache::get_memory(std::vector<int> const& dims,
                                     mkldnn::memory::format format) {
                {
                    // search matched memory
                    auto found = std::find_if(
                      cached_memory_list_.begin(), cached_memory_list_.end(),
                      [&dims, format](auto const& m) {
                          return extract_dims(m) == dims &&
                                 extract_format(m) == format;
                      });

                    // when found
                    if(found != cached_memory_list_.end()) {
                        return std::make_tuple(*found, nullopt);
                    }
                }

                // try to convert array to memory
                if(original_array_ &&
                   (ndims_to_data_memory_format(dims.size()) == format ||
                    ndims_to_weight_memory_format(dims.size()) == format)) {
                    mkldnn::memory new_memory(
                      {{{dims},
                        dtype_to_mkldnn_memory_data_type(
                          original_array_->dtype()),
                        format},
                       engine()},
                      const_cast<void*>(original_array_->data()));
                    add_cached_memory(new_memory);
                    return std::make_tuple(new_memory, nullopt);
                }

                // when not found matched memory
                {
                    // search same dims and format type (data or weight)
                    // memory
                    auto found = std::find_if(
                      cached_memory_list_.begin(), cached_memory_list_.end(),
                      [&dims, format](auto const& m) {
                          return extract_dims(m) == dims &&
                                 is_data_format(extract_format(m)) ==
                                   is_data_format(format);
                      });

                    // when found same dims and format type memory
                    if(found != cached_memory_list_.end()) {
                        auto const& found_memory = *found;
                        assert(dims == extract_dims(found_memory));
                        mkldnn::memory new_memory(
                          {{{dims}, extract_data_type(found_memory), format},
                           engine()});
                        auto reorder_primitive =
                          mkldnn::reorder(found_memory,
                                          new_memory);
                        add_cached_memory(new_memory);
                        return std::make_tuple(new_memory, reorder_primitive);
                    }
                }

                if(!original_array_) {
                    throw std::runtime_error("not found valid array or memory");
                }

                // when not found same dims and format type memory
                assert(original_array_);
                assert(total_size(*original_array_) ==
                       std::accumulate(dims.begin(), dims.end(), 1,
                                       std::multiplies<int>()));

                auto original_dtype =
                  dtype_to_mkldnn_memory_data_type(original_array_->dtype());
                mkldnn::memory base_memory(
                  {{{dims},
                    original_dtype,
                    is_data_format(format)
                      ? ndims_to_data_memory_format(dims.size())
                      : ndims_to_weight_memory_format(dims.size())},
                   engine()},
                  const_cast<void*>(original_array_->data()));
                add_cached_memory(base_memory);

                mkldnn::memory new_memory(
                  {{{dims}, original_dtype, format}, engine()});
                add_cached_memory(new_memory);

                auto reorder_primitive =
                  mkldnn::reorder(base_memory, new_memory);
                return std::make_tuple(new_memory, reorder_primitive);
            }

            mkldnn::memory memory_cache::get_data_memory() {
                auto found =
                  std::find_if(cached_memory_list_.begin(),
                               cached_memory_list_.end(), [](auto const& m) {
                                   return is_data_format(extract_format(m));
                               });

                // when found data format type memory
                if(found != cached_memory_list_.end()) {
                    return *found;
                }

                assert(original_array_);
                auto original_dtype =
                  dtype_to_mkldnn_memory_data_type(original_array_->dtype());
                mkldnn::memory base_memory(
                  {{{dims()},
                    original_dtype,
                    ndims_to_data_memory_format(dims().size())},
                   engine()},
                  const_cast<void*>(original_array_->data()));
                add_cached_memory(base_memory);
                return base_memory;
            }

        } // namespace mkldnn_backend
    }     // namespace mkldnn_with_generic_fallback_backend
} // namespace menoh_impl
