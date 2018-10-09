#include <gtest/gtest.h>

#include <menoh/attribute_completion_and_shape_inference.hpp>
#include <menoh/model_data.hpp>

#include "common.hpp"
namespace {

    class AttributeCompletionAndShapeInferenceTest : public ::testing::Test {};

    TEST_F(AttributeCompletionAndShapeInferenceTest, conv_completion) {
        menoh_impl::model_data model_data;
        model_data.node_list.push_back(
          menoh_impl::node{"Conv", {"x", "w"}, {"y"}, {}});
        std::unordered_map<std::string, menoh_impl::array_profile>
          input_profile_table;
        input_profile_table.emplace(
          "x",
          menoh_impl::array_profile(menoh_impl::dtype_t::float_, {1, 1, 3, 3}));
        input_profile_table.emplace(
          "w",
          menoh_impl::array_profile(menoh_impl::dtype_t::float_, {1, 1, 3, 3}));
        menoh_impl::complete_attribute_and_infer_shape(model_data,
                                                       input_profile_table);
        auto const& node = model_data.node_list.at(0);
        auto found = node.attribute_table.find("kernel_shape");
        ASSERT_NE(found, node.attribute_table.end());
        menoh_impl::assert_eq_list(
          menoh_impl::get<std::vector<int>>(found->second),
          std::vector<int>({3, 3}));
    }

    TEST_F(AttributeCompletionAndShapeInferenceTest, sum_check) {
        menoh_impl::model_data model_data;
        model_data.node_list.push_back(
          menoh_impl::node{"Sum", {"x0", "x1", "x2"}, {"y"}, {}});
        std::unordered_map<std::string, menoh_impl::array_profile>
          input_profile_table;
        for(int i = 0; i < 3; ++i) {
            input_profile_table.emplace(
              "x" + std::to_string(i),
              menoh_impl::array_profile(menoh_impl::dtype_t::float_,
                                        {1, 1, 3, 3}));
        }
        ASSERT_NO_THROW(menoh_impl::complete_attribute_and_infer_shape(
          model_data, input_profile_table));
    }

    TEST_F(AttributeCompletionAndShapeInferenceTest, add_check) {
        menoh_impl::model_data model_data;
        model_data.node_list.push_back(
          menoh_impl::node{"Add", {"x0", "x1"}, {"y"}, {}});
        std::unordered_map<std::string, menoh_impl::array_profile>
          input_profile_table;
        for(int i = 0; i < 2; ++i) {
            input_profile_table.emplace(
              "x" + std::to_string(i),
              menoh_impl::array_profile(menoh_impl::dtype_t::float_,
                                        {1, 1, 3, 3}));
        }
        ASSERT_NO_THROW(menoh_impl::complete_attribute_and_infer_shape(
          model_data, input_profile_table));
    }

} // namespace
