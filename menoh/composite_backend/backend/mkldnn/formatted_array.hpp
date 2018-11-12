#ifndef MENOH_IMPL_COMPOSITE_BACKEND_MKLDNN_FORMATTED_ARRAY_HPP
#define MENOH_IMPL_COMPOSITE_BACKEND_MKLDNN_FORMATTED_ARRAY_HPP

#include <menoh/array.hpp>

#include <menoh/composite_backend/backend/mkldnn/memory_conversion.hpp>

#include <mkldnn.hpp>

namespace menoh_impl {
    namespace composite_backend {
        namespace mkldnn_backend {

            class formatted_array {
            public:
                formatted_array() = default;

                formatted_array(mkldnn::memory::format format, array const& arr)
                  : format_(format), array_(arr) {}

                mkldnn::memory make_memory(mkldnn::memory::format format,
                                           mkldnn::engine const& engine) const {
                    return mkldnn::memory(
                      {{{array_.dims()},
                        dtype_to_mkldnn_memory_data_type(array_.dtype()),
                        format},
                       engine},
                      const_cast<void*>(array_.data()));
                }

                mkldnn::memory::format format() const { return format_; }
                menoh_impl::array array() const { return array_; }

            private:
                mkldnn::memory::format format_;
                menoh_impl::array array_;
            };

            inline bool is_format_any(formatted_array const& farr) {
                return farr.format() == mkldnn::memory::format::any;
            }

            inline mkldnn::memory
            make_memory(formatted_array const& farr,
                        mkldnn::engine const& engine) {
                assert(farr.format() != mkldnn::memory::format::any);
                return farr.make_memory(farr.format(), engine);
            }

        } // namespace mkldnn_backend
    }     // namespace composite_backend
} // namespace menoh_impl

#endif // MENOH_IMPL_COMPOSITE_BACKEND_MKLDNN_FORMATTED_ARRAY_HPP
