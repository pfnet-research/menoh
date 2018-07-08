#ifndef MENOH_NAIVE_COMPUTATION_FACTORY_HPP
#define MENOH_NAIVE_COMPUTATION_FACTORY_HPP

#include <functional>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <menoh/array.hpp>
#include <menoh/naive/computation_node.hpp>

namespace menoh_impl {
    namespace naive_backend {

        using computation_node_factory_return_type =
          std::tuple<std::function<void()>,
                     std::vector<std::pair<std::string, array>>>;

        using computation_node_factory =
          std::function<computation_node_factory_return_type(
            int32_t, std::vector<node> const&,
            std::unordered_map<std::string, array> const&)>;

    } // namespace naive_backend
} // namespace menoh_impl

#endif // MENOH_NAIVE_COMPUTATION_FACTORY_HPP
