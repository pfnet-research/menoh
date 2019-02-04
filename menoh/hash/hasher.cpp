#include <menoh/hash/hasher.hpp>

#include <iomanip>

namespace menoh_impl {

    hasher::hasher() {
        constexpr auto seed = 0;
        menoh_XXH64_reset(state_, seed);
    }

    void hasher::add(const std::uint8_t* data, std::size_t size) {
        menoh_XXH64_update(state_, data, size);
    }

    std::string hasher::finish() {
        std::uint64_t hash = menoh_XXH64_digest(state_);
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(16) << std::hex << hash;
        return ss.str();
    }

} // namespace menoh_impl
