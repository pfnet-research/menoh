#ifndef MENOH_NAIVE_OPERATOR_INDEX_CONVERSION_HPP
#define MENOH_NAIVE_OPERATOR_INDEX_CONVERSION_HPP

namespace menoh_impl {
    namespace naive_backend {

        class index_converter {
        public:
            explicit index_converter(std::vector<int32_t> const& dims) {
                int32_t offset = 1;
                for(auto di = std::rbegin(dims); di != std::rend(dims); ++di) {
                    offsets_.push_back(offset);
                    offset *= *di;
                }
                std::reverse(offsets_.begin(), offsets_.end());
            }

            std::vector<int32_t>
            flat_index_to_indices(int32_t flat_index) const {
                std::vector<int32_t> indices;
                for(auto offset : offsets_) {
                    indices.push_back(flat_index / offset);
                    flat_index %= offset;
                }
                return indices;
            }

            int32_t
            indices_to_flat_index(std::vector<int32_t> const& indices) const {
                assert(indices.size() == offsets_.size());
                int32_t flat_index = 0;
                for(int32_t i = 0; i < static_cast<int32_t>(indices.size());
                    ++i) {
                    flat_index += indices[i] * offsets_[i];
                }
                return flat_index;
            }

        private:
            std::vector<int32_t> offsets_;
        };

    } // namespace naive_backend
} // namespace menoh_impl

#endif // MENOH_NAIVE_OPERATOR_INDEX_CONVERSION_HPP
