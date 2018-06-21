#ifndef MENOH_TEST_NP_IO_HPP
#define MENOH_TEST_NP_IO_HPP

#include <fstream>
#include <numeric>
#include <string>
#include <tuple>

#include <menoh/array.hpp>

namespace menoh_impl {
    inline auto load_np_array(std::string const& filename) {
        std::ifstream ifs(filename);
        if(!ifs) {
            throw std::runtime_error(filename + " is not found");
        }
        std::string line;
        std::getline(ifs, line);
        auto dims_num = std::stoi(line);
        std::vector<int> shape;
        {
            std::getline(ifs, line);
            std::istringstream iss(line);
            for(int i = 0; i < dims_num; ++i) {
                int n;
                iss >> n;
                shape.push_back(n);
            }
        }

        std::vector<float> data;
        {
            auto total_num = std::accumulate(shape.begin(), shape.end(), 1,
                                             std::multiplies<>());
            std::getline(ifs, line);
            std::istringstream iss(line);
            for(int i = 0; i < total_num; ++i) {
                float f;
                iss >> f;
                data.push_back(f);
            }
        }
        return std::make_tuple(dims_num, shape, data);
    }

    inline auto load_np_array_as_array(std::string const& filename) {
        int dims_num;
        std::vector<int> shape;
        std::vector<float> data;
        std::tie(dims_num, shape, data) = menoh_impl::load_np_array(filename);
        array arr(dtype_t::float_, shape);
        std::copy(data.begin(), data.end(), fbegin(arr));
        return arr;
    }
} // namespace menoh_impl

#endif // MENOH_TEST_NP_IO_HPP
