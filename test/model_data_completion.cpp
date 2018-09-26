#include <gtest/gtest.h>

#include <menoh/model_completion.hpp>
#include <menoh/model_data.hpp>

#include "common.hpp"
namespace {

    class ModelDataCompletionTest : public ::testing::Test {};

    TEST_F(ModelDataCompletionTest, completion) {
        menoh_impl::model_data model_data;
        model_data.node_list.push_back(
          menoh_impl::node{"Conv", {"x", "w"}, {"y"}, {}});
        std::unordered_map<std::string, menoh_impl::array_profile>
          input_profile_table;
        input_profile_table.emplace(
          "w",
          menoh_impl::array_profile(menoh_impl::dtype_t::float_, {1, 1, 3, 3}));
        menoh_impl::complete_model_data(model_data, input_profile_table);
        auto const& node = model_data.node_list.at(0);
        auto found = node.attribute_table.find("kernel_shape");
        ASSERT_NE(found, node.attribute_table.end());
        menoh_impl::assert_eq_list(
          menoh_impl::get<std::vector<int>>(found->second),
          std::vector<int>({3, 3}));
    }

} // namespace
