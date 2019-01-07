
#if defined(_WIN32) || defined(__WIN32__)
#include <Windows.h>
#endif

#include <assert.h>
#include <stdio.h>

#include <menoh/menoh.h>

#define ERROR_CHECK(statement)                            \
    {                                                     \
        menoh_error_code ec = statement;                  \
        if(ec) {                                          \
            printf("%s", menoh_get_last_error_message()); \
            return 0;                                     \
        }                                                 \
    }

int main() {
    menoh_error_code ec = menoh_error_code_success;

    const char* conv1_1_in_name = "Input_0";
    const char* fc6_out_name = "Gemm_0";
    const char* softmax_out_name = "Softmax_0";

    menoh_model_data_handle model_data;
    ERROR_CHECK(
      menoh_make_model_data_from_onnx("../data/vgg16.onnx", &model_data));

    menoh_variable_profile_table_builder_handle vpt_builder;
    ERROR_CHECK(menoh_make_variable_profile_table_builder(&vpt_builder));

    const int32_t input_dims[] = {1, 3, 224, 224};
    ERROR_CHECK(menoh_variable_profile_table_builder_add_input_profile(
      vpt_builder, conv1_1_in_name, menoh_dtype_float, 4, input_dims));

    ERROR_CHECK(menoh_variable_profile_table_builder_add_output_name(
      vpt_builder, fc6_out_name));
    ERROR_CHECK(menoh_variable_profile_table_builder_add_output_name(
      vpt_builder, softmax_out_name));

    menoh_variable_profile_table_handle variable_profile_table;
    ERROR_CHECK(menoh_build_variable_profile_table(vpt_builder, model_data,
                                                   &variable_profile_table));

    /*
    int32_t softmax_out_dims[2];
    ERROR_CHECK(menoh_variable_profile_table_get_dims_at(
      variable_profile_table, softmax_out_name, 0, &softmax_out_dims[0]));
    ERROR_CHECK(menoh_variable_profile_table_get_dims_at(
      variable_profile_table, softmax_out_name, 1, &softmax_out_dims[1]));
    */

    int32_t softmax_out_dims_size;
    const int32_t* softmax_out_dims;
    ERROR_CHECK(menoh_variable_profile_table_get_dims(
      variable_profile_table, softmax_out_name, &softmax_out_dims_size,
      &softmax_out_dims));
    assert(softmax_out_dims_size == 1);
    assert(softmax_out_dims[0]== 2);
    assert(softmax_out_dims[1] == 1000);

    ERROR_CHECK(menoh_model_data_optimize(model_data, variable_profile_table));

    menoh_model_builder_handle model_builder;
    ERROR_CHECK(
      menoh_make_model_builder(variable_profile_table, &model_builder));

    float input_buff[1 * 3 * 224 * 224];
    menoh_model_builder_attach_external_buffer(model_builder, conv1_1_in_name,
                                               input_buff);

    menoh_model_handle model;
    ERROR_CHECK(
      menoh_build_model(model_builder, model_data, "mkldnn", "", &model));

    menoh_delete_model_data(
      model_data); // you can delete model_data after model building

    float* fc6_output_buff;
    ERROR_CHECK(menoh_model_get_variable_buffer_handle(model, fc6_out_name,
                                              (void**)&fc6_output_buff));
    float* softmax_output_buff;
    ERROR_CHECK(menoh_model_get_variable_buffer_handle(model, softmax_out_name,
                                              (void**)&softmax_output_buff));

    // initialie input_buf here
    for(int i = 0; i < 1 * 3 * 224 * 224; ++i) {
        input_buff[i] = 0.5;
    }

    ERROR_CHECK(menoh_model_run(model));

    for(int i = 0; i < 10; ++i) {
        printf("%f ", *(fc6_output_buff + i));
    }
    printf("\n");
    for(int n = 0; n < softmax_out_dims[0]; ++n) {
        for(int i = 0; i < softmax_out_dims[1]; ++i) {
            printf("%f ", *(softmax_output_buff + n * softmax_out_dims[1] + i));
        }
        printf("\n");
    }

    menoh_delete_model(model);
    menoh_delete_model_builder(model_builder);
    menoh_delete_variable_profile_table_builder(vpt_builder);
}
