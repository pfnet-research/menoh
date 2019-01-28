#ifndef MENOH_HASH_HASH_HPP
#define MENOH_HASH_HASH_HPP

#include <cstdint>
#include <string>

#include <menoh/hash/xxhash.h>

namespace menoh_impl {

    class hasher {
    public:
        hasher();

        void add(const std::uint8_t* data, std::size_t size);

        std::string finish();

    private:
        XXH64_state_t* const state_ = menoh_XXH64_createState();
    };

    inline void add_str(hasher& h, std::string const& s) {
        h.add(static_cast<const uint8_t*>(static_cast<const void*>(s.data())),
              s.size());
    };

} // namespace menoh_impl

#endif // MENOH_HASH_HASH_HPP
