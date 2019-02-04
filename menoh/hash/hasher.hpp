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

    template <typename Container>
    void add_container(hasher& h, Container const& c) {
        h.add(static_cast<const uint8_t*>(static_cast<const void*>(c.data())),
              c.size());
    };

    inline
    void add_c_str(hasher& h, const char* c_str) {
        add_container(h, std::string(c_str));
    }

} // namespace menoh_impl

#endif // MENOH_HASH_HASH_HPP
