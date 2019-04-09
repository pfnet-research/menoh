#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <numeric>

#include <filesystem/path.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx/onnx_pb.h>

#include <menoh/menoh.hpp>

#include "common.hpp"

namespace {

    struct named_array_data {
        std::string name;
        menoh::dtype_t dtype;
        std::vector<int> dims;
        std::unique_ptr<char[]> data;
    };

    auto load_param(filesystem::path const& filepath,
                    bool squash_dims = false) {
        namespace gpio = ::google::protobuf::io;

        std::ifstream ifs(filepath.str(), std::ios::binary);
        if(!ifs) {
            std::cout << "invalid filename" << std::endl;
            throw "invalid_filename";
        }
        gpio::IstreamInputStream iis(&ifs);
        gpio::CodedInputStream cis(&iis);
        cis.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                               std::numeric_limits<int>::max());
        menoh_onnx::TensorProto tensor;
        if(!tensor.ParseFromCodedStream(&cis)) {
            std::cout << "invalid filename" << std::endl;
            throw "onnx_parse_error";
        }

        // TODO int array
        std::vector<int> dims;
        for(auto d : tensor.dims()) {
            dims.push_back(static_cast<int>(d));
        }
        //tensor.dims().begin(), tensor.dims().end());
        std::cout << "dims (";
        for(auto d : dims)  {
            std::cout << d << " ";
        }
        std::cout << ")" << std::endl;
        if(squash_dims) {
            assert(2 <= dims.size());
            dims.at(1) = std::accumulate(dims.begin() + 1, dims.end(), 1,
                                         std::multiplies<int>());
            dims.erase(dims.begin() + 2, dims.end());
        }
        auto total_size =
          std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
        std::cout << "total_size " << total_size << std::endl;
        assert(tensor.has_raw_data());
        if(tensor.data_type() == menoh_onnx::TensorProto_DataType_FLOAT) {
            assert(tensor.raw_data().length() ==
                   static_cast<decltype(tensor.raw_data().length())>(
                     total_size * 4));

            auto data = std::make_unique<char[]>(total_size * 4);
            std::copy(tensor.raw_data().begin(), tensor.raw_data().end(),
                      data.get());
            // TODO other dtype
            return named_array_data{tensor.name(), menoh::dtype_t::float32,
                                    std::move(dims), std::move(data)};
        }
        if(tensor.data_type() == menoh_onnx::TensorProto_DataType_INT64) {
            assert(tensor.raw_data().length() ==
                   static_cast<decltype(tensor.raw_data().length())>(
                     total_size * 8));

            auto data = std::make_unique<char[]>(total_size * 8);
            std::copy(tensor.raw_data().begin(), tensor.raw_data().end(),
                      data.get());
            // TODO other dtype
            return named_array_data{tensor.name(), menoh::dtype_t::int64,
                                    std::move(dims), std::move(data)};
        }
        throw "unexpected tensor data type";
    }

    class OperatorTest : public ::testing::Test {
    protected:
        OperatorTest()
          : onnx_test_data_dir_path_(
              "../external/onnx/onnx/backend/test/data/node/") {}

        void run_test(std::string backend_name, std::string const& test_name,
                      float eps, bool squash_dims = false, bool static_params = false) {
            auto parent_dir_path = onnx_test_data_dir_path_ / test_name;

            for(int data_set_index = 0; true; ++data_set_index) {
                auto dataset_path =
                  parent_dir_path /
                  ("test_data_set_" + std::to_string(data_set_index));
                if(!dataset_path.exists()) {
                    break;
                }
                std::vector<named_array_data> input_list;
                for(int input_index = 0;; ++input_index) {
                    auto input_data_path =
                      dataset_path /
                      ("input_" + std::to_string(input_index) + ".pb");
                    if(!input_data_path.exists()) {
                        break;
                    }
                    input_list.push_back(
                      load_param(input_data_path, squash_dims));
                }
                std::vector<named_array_data> true_output_list;
                for(int output_index = 0;; ++output_index) {
                    auto output_data_path =
                      dataset_path /
                      ("output_" + std::to_string(output_index) + ".pb");
                    if(!output_data_path.exists()) {
                        break;
                    }
                    true_output_list.push_back(
                      load_param(output_data_path, squash_dims));
                }

                auto onnx_model_filename = parent_dir_path / "model.onnx";
                auto model_data =
                  menoh::make_model_data_from_onnx(onnx_model_filename.str());
                menoh::variable_profile_table_builder vpt_builder;

                if(static_params) {
                    vpt_builder.add_input_profile(input_list.front().name, input_list.front().dtype,
                                                  input_list.front().dims);
                    for(std::size_t i = 1; i < input_list.size(); ++i) {
                        auto& input = input_list.at(i);
                        menoh_model_data_add_parameter(model_data.get(), input.name.c_str(),
                                static_cast<menoh_dtype>(input.dtype), input.dims.size(),
                                input.dims.data(), input.data.get());
                    }
                }
                else {
                    for(auto const& input : input_list) {
                        vpt_builder.add_input_profile(input.name, input.dtype,
                                                      input.dims);
                    }
                }
                for(auto const& output : true_output_list) {
                    vpt_builder.add_output_name(output.name);
                }
                auto vpt = vpt_builder.build_variable_profile_table(model_data);
                menoh::model_builder model_builder(vpt);

                if(static_params) {
                    model_builder.attach_external_buffer(
                      input_list.front().name, static_cast<void*>(input_list.front().data.get()));
                }
                else {
                    for(auto const& input : input_list) {
                        model_builder.attach_external_buffer(
                          input.name, static_cast<void*>(input.data.get()));
                    }
                }

                auto model = model_builder.build_model(
                  model_data, backend_name, R"({"log_output" : "stdout"})");

                model_data.reset();

                std::vector<menoh::variable> output_list;
                for(auto const& true_output : true_output_list) {
                    output_list.push_back(model.get_variable(true_output.name));
                }
                model.run();
                assert(true_output_list.size() == output_list.size());
                auto static_cast_to_float_ptr = [](auto p) {
                    return static_cast<float*>(static_cast<void*>(p));
                };
                for(unsigned int output_index = 0;
                    output_index < true_output_list.size(); ++output_index) {
                    auto const& input = input_list.front();
                    static_cast<void>(input); // maybe unused
                    auto const& output = output_list.at(output_index);
                    auto const& true_output = true_output_list.at(output_index);
                    auto total_size = std::accumulate(true_output.dims.begin(),
                                                      true_output.dims.end(), 1,
                                                      std::multiplies<int>());
                    /*
                    std::cout << true_output.name << std::endl;
                    for(auto i = 0; i < 10; ++i) {
                        std::cout
                          << *(static_cast<float*>(
                                 static_cast<void*>(input.data.get())) +
                               i)
                          << " "
                          << *(static_cast<float*>(output.buffer_handle) + i)
                          << " "
                          << *(static_cast<float*>(
                                 static_cast<void*>(true_output.data.get())) +
                               i)
                          << std::endl;
                    }
                    */
                    menoh_impl::assert_eq_list(
                      static_cast<float*>(output.buffer_handle),
                      static_cast<float*>(output.buffer_handle) + total_size,
                      static_cast_to_float_ptr(true_output.data.get()),
                      static_cast_to_float_ptr(true_output.data.get()) +
                        total_size,
                      eps);
                }
            }
        }

    private:
        filesystem::path onnx_test_data_dir_path_;
    };

#define TEST_OP_IMPL(backend_name, test_name, eps, squash, static_params) \
    TEST_F(OperatorTest, backend_name##_##test_name) {     \
        run_test(#backend_name, #test_name, eps, squash, static_params);  \
    }
#define TEST_OP(backend_name, test_name, eps) \
    TEST_OP_IMPL(backend_name, test_name, eps, false, false)
#define TEST_OP_SQUASH_DIMS(backend_name, test_name, eps) \
    TEST_OP_IMPL(backend_name, test_name, eps, true, false)
#define TEST_OP_STATIC_PARAMS(backend_name, test_name, eps) \
    TEST_OP_IMPL(backend_name, test_name, eps, false, true)

    float eps = 1.e-4;

#ifdef MENOH_WITH_MKLDNN
    // Tests for MKLDNN backend
    TEST_OP_SQUASH_DIMS(mkldnn, test_abs, eps);
    TEST_OP_SQUASH_DIMS(mkldnn, test_elu, eps);
    TEST_OP_SQUASH_DIMS(mkldnn, test_elu_default, eps);
    TEST_OP_SQUASH_DIMS(mkldnn, test_leakyrelu, eps);
    TEST_OP_SQUASH_DIMS(mkldnn, test_leakyrelu_default, eps);
    TEST_OP_SQUASH_DIMS(mkldnn, test_relu, eps);
    TEST_OP_SQUASH_DIMS(mkldnn, test_sqrt, eps);
    TEST_OP_SQUASH_DIMS(mkldnn, test_tanh, eps);

    TEST_OP(mkldnn, test_averagepool_2d_default, eps);
    TEST_OP_SQUASH_DIMS(mkldnn, test_add, eps);
    // TEST_OP_SQUASH_DIMS(mkldnn, test_batchnormalization, eps); // not found
    // TEST_OP(mkldnn, test_concat_2d_axis_0, eps);
    // TEST_OP(mkldnn, test_concat_2d_axis_1, eps);
    // TEST_OP(mkldnn, test_conv_with_strides_padding, eps);
    // TEST_OP_SQUASH_DIMS(mkldnn, test_convtranspose, eps); // not found
    // TEST_OP(mkldnn, test_gemm_nobroadcast, eps);
    TEST_OP(mkldnn, test_globalaveragepool, eps);
    TEST_OP(mkldnn, test_globalaveragepool_precomputed, eps);
    TEST_OP(mkldnn, test_globalmaxpool, eps);
    TEST_OP(mkldnn, test_globalmaxpool_precomputed, eps);
    // TEST_OP(mkldnn, test_globalaveragepool, eps);
    // TEST_OP(mkldnn, test_globalmaxpool, eps);
    TEST_OP(mkldnn, test_maxpool_2d_default, eps);
    TEST_OP_SQUASH_DIMS(mkldnn, test_softmax_axis_1, eps);
    // TEST_OP_SQUASH_DIMS(mkldnn, test_sum_one_input, eps);
    // TEST_OP_SQUASH_DIMS(mkldnn, test_sum_two_inputs, eps);

    // TEST_OP(mkldnn, test_averagepool_2d_pads, eps);
    // TEST_OP(mkldnn, test_averagepool_2d_precomputed_pads, eps);
    // TEST_OP(mkldnn, test_averagepool_2d_precomputed_same_upper, eps);


    // Tests for MKLDNN with Generic fallback backend

    // BatchNormalization
    TEST_OP(mkldnn_with_generic_fallback, test_batchnorm_epsilon, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_batchnorm_example, eps);
  
    // Conv
    TEST_OP(mkldnn_with_generic_fallback, test_basic_conv_without_padding, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_basic_conv_with_padding, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_conv_with_strides_and_asymmetric_padding, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_conv_with_strides_no_padding, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_conv_with_strides_padding, eps);

    TEST_OP(mkldnn_with_generic_fallback, test_constant, eps);
  
    // Eltwise
    TEST_OP_SQUASH_DIMS(mkldnn_with_generic_fallback, test_abs, eps);
    TEST_OP_SQUASH_DIMS(mkldnn_with_generic_fallback, test_elu, eps);
    TEST_OP_SQUASH_DIMS(mkldnn_with_generic_fallback, test_leakyrelu, eps);
    TEST_OP_SQUASH_DIMS(mkldnn_with_generic_fallback, test_relu, eps);
    TEST_OP_SQUASH_DIMS(mkldnn_with_generic_fallback, test_sqrt, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_sigmoid, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_sigmoid_example, eps);
    TEST_OP_SQUASH_DIMS(mkldnn_with_generic_fallback, test_tanh, eps);

    //TEST_OP(mkldnn_with_generic_fallback, test_gemm_nobroadcast, eps);

    // Mul
    TEST_OP(mkldnn_with_generic_fallback, test_mul, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_mul_bcast, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_mul_example, eps);
  
    // Pool
    //TEST_OP(mkldnn_with_generic_fallback, test_averagepool_1d_default, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_averagepool_2d_default, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_averagepool_2d_pads, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_averagepool_2d_pads_count_include_pad, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_averagepool_2d_precomputed_pads, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_averagepool_2d_precomputed_pads_count_include_pad, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_averagepool_2d_precomputed_same_upper, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_averagepool_2d_precomputed_strides, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_averagepool_2d_same_lower, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_averagepool_2d_same_upper, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_averagepool_2d_strides, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_averagepool_3d_default, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_maxpool_1d_default, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_maxpool_2d_default, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_maxpool_2d_pads, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_maxpool_2d_precomputed_pads, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_maxpool_2d_precomputed_same_upper, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_maxpool_2d_precomputed_strides, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_maxpool_2d_same_lower, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_maxpool_2d_same_upper, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_maxpool_2d_strides, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_maxpool_3d_default, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_maxpool_with_argmax_2d_precomputed_pads, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_maxpool_with_argmax_2d_precomputed_strides, eps);

    // Reshape
    //TEST_OP(mkldnn_with_generic_fallback, test_reshape_extended_dims, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_reshape_negative_dim, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_reshape_one_dim, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_reshape_reduced_dims, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_reshape_reordered_dims, eps);

    // Softmax
    //TEST_OP(mkldnn_with_generic_fallback, test_softmax_axis_0, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_softmax_axis_1, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_softmax_axis_2, eps);
    //TEST_OP(mkldnn_with_generic_fallback, test_softmax_default_axis, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_softmax_example, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_softmax_large_number, eps);

    // Sum and Add
    TEST_OP(mkldnn_with_generic_fallback, test_sum_example, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_sum_one_input, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_sum_two_inputs, eps);
    // TEST_OP(mkldnn_with_generic_fallback, test_add, eps); // ndims=3 is not
    // implemented yet (mkldnn will support soon)
    // TEST_OP(mkldnn_with_generic_fallback, test_add_bcast, eps); //broadcast
    // is not implemented yet

    // Transpose
    TEST_OP(mkldnn_with_generic_fallback, test_transpose_all_permutations_0, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_transpose_all_permutations_1, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_transpose_all_permutations_2, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_transpose_all_permutations_3, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_transpose_all_permutations_4, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_transpose_all_permutations_5, eps);
    TEST_OP(mkldnn_with_generic_fallback, test_transpose_default, eps);

#endif // MENOH_WITH_MKLDNN

#ifdef MENOH_WITH_TENSORRT
    // BatchNormalization
    TEST_OP_STATIC_PARAMS(tensorrt, test_batchnorm_epsilon, eps); // fails
    TEST_OP_STATIC_PARAMS(tensorrt, test_batchnorm_example, eps);

    // Concat
    //TEST_OP_STATIC_PARAMS(tensorrt, test_concat_1d_axis_0, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_concat_2d_axis_0, eps);
    TEST_OP(tensorrt, test_concat_2d_axis_1, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_concat_3d_axis_0, eps);
    TEST_OP(tensorrt, test_concat_3d_axis_1, eps);
    TEST_OP(tensorrt, test_concat_3d_axis_2, eps);

    // Const
    TEST_OP_STATIC_PARAMS(tensorrt, test_const, eps);
  
    // Conv
    TEST_OP_STATIC_PARAMS(tensorrt, test_basic_conv_without_padding, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_basic_conv_with_padding, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_conv_with_strides_and_asymmetric_padding, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_conv_with_strides_no_padding, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_conv_with_strides_padding, eps);

    // Eltwise
    //TEST_OP_STATIC_PARAMS(tensorrt, test_abs, eps); // not supported
    //TEST_OP_STATIC_PARAMS(tensorrt, test_elu, eps);  // not supported
    //TEST_OP_STATIC_PARAMS(tensorrt, test_leakyrelu, eps); // not supported
    TEST_OP_STATIC_PARAMS(tensorrt, test_relu, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_sqrt, eps); // not supported
    TEST_OP_STATIC_PARAMS(tensorrt, test_sigmoid, eps); // not supported
    TEST_OP_STATIC_PARAMS(tensorrt, test_sigmoid_example, eps); // not supported
    TEST_OP_STATIC_PARAMS(tensorrt, test_tanh, eps);

    // Gemm
    //TEST_OP_STATIC_PARAMS(tensorrt, test_gemm_broadcast, eps);   // not support (need transpose)
    //TEST_OP_STATIC_PARAMS(tensorrt, test_gemm_nobroadcast, eps); // not support (need transpose)

    // Identity
    TEST_OP_STATIC_PARAMS(tensorrt, test_identity, eps);

    // LRN
    //TEST_OP_STATIC_PARAMS(tensorrt, test_lrn, tolerant_eps); // not support
    //TEST_OP_STATIC_PARAMS(tensorrt, test_lrn_default, tolerant_eps); // not support
    
    // Mul
    TEST_OP(tensorrt, test_mul, eps);
    // TEST_OP_STATIC_PARAMS(tensorrt, test_mul_bcast, eps);  // not support broadcasting (TODO implement by TensorRT's Scale function)
    //TEST_OP(tensorrt, test_mul_example, eps); // not support 1d input

    // Softmax
    //TEST_OP_STATIC_PARAMS(tensorrt, test_softmax_axis_0, eps); // not support axis!=1
    TEST_OP_STATIC_PARAMS(tensorrt, test_softmax_axis_1, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_softmax_axis_2, eps); // not support axis!=1
    TEST_OP_STATIC_PARAMS(tensorrt, test_softmax_default_axis, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_softmax_example, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_softmax_large_number, eps);

    // Pool
    //TEST_OP_STATIC_PARAMS(tensorrt, test_averagepool_1d_default, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_averagepool_2d_default, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_averagepool_2d_pads, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_averagepool_2d_pads_count_include_pad, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_averagepool_2d_precomputed_pads, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_averagepool_2d_precomputed_pads_count_include_pad, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_averagepool_2d_precomputed_same_upper, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_averagepool_2d_precomputed_strides, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_averagepool_2d_same_lower, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_averagepool_2d_same_upper, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_averagepool_2d_strides, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_averagepool_3d_default, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_maxpool_1d_default, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_maxpool_2d_default, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_maxpool_2d_pads, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_maxpool_2d_precomputed_pads, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_maxpool_2d_precomputed_same_upper, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_maxpool_2d_precomputed_strides, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_maxpool_2d_same_lower, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_maxpool_2d_same_upper, eps);
    TEST_OP_STATIC_PARAMS(tensorrt, test_maxpool_2d_strides, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_maxpool_3d_default, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_maxpool_with_argmax_2d_precomputed_pads, eps); // not found z
    //TEST_OP_STATIC_PARAMS(tensorrt, test_maxpool_with_argmax_2d_precomputed_strides, eps); // not found z

    // Reshape
    //TEST_OP_STATIC_PARAMS(tensorrt, test_reshape_extended_dims, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_reshape_negative_dim, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_reshape_one_dim, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_reshape_reduced_dims, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_reshape_reordered_dims, eps);

    // Sum and Add
    //TEST_OP_STATIC_PARAMS(tensorrt, test_sum_example, eps); // not support 1d input
    //TEST_OP_STATIC_PARAMS(tensorrt, test_sum_one_input, eps); // not support 1d input
    //TEST_OP_STATIC_PARAMS(tensorrt, test_sum_two_inputs, eps); // not support 1d input
    TEST_OP(tensorrt, test_add, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_add_bcast, eps); //broadcast is not implemented yet

    // Transpose // not supported
    //TEST_OP_STATIC_PARAMS(tensorrt, test_transpose_all_permutations_0, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_transpose_all_permutations_1, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_transpose_all_permutations_2, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_transpose_all_permutations_3, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_transpose_all_permutations_4, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_transpose_all_permutations_5, eps);
    //TEST_OP_STATIC_PARAMS(tensorrt, test_transpose_default, eps);
    
    TEST_OP_STATIC_PARAMS(tensorrt, test_unsqueeze, eps);
#endif // MENOH_WITH_TENSORRT

#undef TEST_OP_SQUASH_DIMS
#undef TEST_OP
#undef TEST_OP_IMPL

} // namespace
