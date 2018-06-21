#ifndef MENOH_C_INTERFACE_H
#define MENOH_C_INTERFACE_H

#include <stdint.h>

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#define MENOH_API __declspec(dllexport)
#else
#define MENOH_API
#endif

#define MENOH_SUPPORTED_ONNX_OPSET_VERSION 7

#ifndef MENOH_ERROR_MESSAGE_MAX_LENGTH
#define MENOH_ERROR_MESSAGE_MAX_LENGTH 1024
#endif

#if MENOH_ERROR_MESSAGE_MAX_LENGTH < 1024
#undef MENOH_ERROR_MESSAGE_MAX_LENGTH
#define MENOH_ERROR_MESSAGE_MAX_LENGTH 1024
#endif

#endif // DOXYGEN_SHOULD_SKIP_THIS

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** @addtogroup c_api C API
 * @{ */

/*! @ingroup vpt
 */
enum menoh_dtype_constant {
    menoh_dtype_float,
};
/*! @ingroup vpt
 */
typedef int32_t menoh_dtype;

/*! @addtogroup error_handling Error handling
 * @{ */
/*! \breaf error_code
 *
 * Users can also get error message via menoh_get_last_error_message().
 */
enum menoh_error_code_constant {
    menoh_error_code_success,
    menoh_error_code_std_error,
    menoh_error_code_unknown_error,
    menoh_error_code_invalid_filename,
    menoh_error_code_unsupported_onnx_opset_version,
    menoh_error_code_onnx_parse_error,
    menoh_error_code_invalid_dtype,
    menoh_error_code_invalid_attribute_type,
    menoh_error_code_unsupported_operator_attribute,
    menoh_error_code_dimension_mismatch,
    menoh_error_code_variable_not_found,
    menoh_error_code_index_out_of_range,
    menoh_error_code_json_parse_error,
    menoh_error_code_invalid_backend_name,
    menoh_error_code_unsupported_operator,
    menoh_error_code_failed_to_configure_operator,
    menoh_error_code_backend_error,
    menoh_error_code_same_named_variable_already_exist,
};
typedef int32_t menoh_error_code;
/*! \brief Users can get detailed message about last error.
 *
 * Users do not need to (and must not) release returned c string.
 */
MENOH_API const char* menoh_get_last_error_message();
/** @} */

/*! @addtogroup model_data Model data types and operations
 * @{ */
/*! \struct menoh_model_data
 * \brief menoh_model_data contains model parameters and computation graph
 * structure.
 */
struct menoh_model_data;
typedef struct menoh_model_data* menoh_model_data_handle;

/*! \brief Load onnx file and make model_data
 */
menoh_error_code MENOH_API menoh_make_model_data_from_onnx(
  const char* onnx_filename, menoh_model_data_handle* dst_handle);
/*! \brief Model_data delete function
 *
 * Users must call to release memory resources allocated for
 * model_data.
 *
 * \note This function can be called after menoh_build_model() function
 * call.
 */
void MENOH_API menoh_delete_model_data(menoh_model_data_handle model_data);
/** @} */

/*! @addtogroup vpt Variable profile table types and operations
 * @{ */
/*! \struct menoh_variable_profile_table_builder
 * \brief menoh_variable_profile_table_builder is the builder for creation of
 * menoh_variable_profile_table.
 *
 * This struct configure profiles of variables.
 *
 * See
 *  - menoh_variable_profile_table_builder_add_input_profile_dims_2()
 *  - menoh_variable_profile_table_builder_add_input_profile_dims_4()
 *  - menoh_variable_profile_table_builder_add_output_profile().
 */
struct menoh_variable_profile_table_builder;
typedef struct menoh_variable_profile_table_builder*
  menoh_variable_profile_table_builder_handle;

/*! \brief Factory function for variable_profile_table_builder
 */
menoh_error_code MENOH_API menoh_make_variable_profile_table_builder(
  menoh_variable_profile_table_builder_handle* dst_handle);
/*! \brief Delete function for variable_profile_table_builder
 *
 * Users must call to release memory resources allocated for
 * variable_profile_table_builder
 */
void MENOH_API menoh_delete_variable_profile_table_builder(
  menoh_variable_profile_table_builder_handle builder);

/*! \brief Add 2D input profile
 *
 * Input profile contains name, dtype and dims (num, size). This 2D input is
 * conventional batched 1D inputs.
 */
menoh_error_code MENOH_API
menoh_variable_profile_table_builder_add_input_profile_dims_2(
  menoh_variable_profile_table_builder_handle builder, const char* name,
  menoh_dtype dtype, int32_t num, int32_t size);

/*! \brief Add 4D input profile
 *
 * Input profile contains name, dtype and dims (num, channel, height, width).
 * This 4D input is conventional batched image inputs. Image input is
 * 3D(channel, height, width).
 */
menoh_error_code MENOH_API
menoh_variable_profile_table_builder_add_input_profile_dims_4(
  menoh_variable_profile_table_builder_handle builder, const char* name,
  menoh_dtype dtype, int32_t num, int32_t channel, int32_t height,
  int32_t width);

/*! \brief Add output profile
 *
 * Output profile contains name and dtype. Its dims are calculated automatically
 * when calling of menoh_build_variable_profile_table.
 */
menoh_error_code MENOH_API
menoh_variable_profile_table_builder_add_output_profile(
  menoh_variable_profile_table_builder_handle builder, const char* name,
  menoh_dtype dtype);

/*! \struct menoh_variable_profile_table
 * \brief menoh_variable_profile_table contains information of dims of
 * variables.
 *
 * Users can access to dims of variables.
 */
struct menoh_variable_profile_table;
typedef struct menoh_variable_profile_table*
  menoh_variable_profile_table_handle;

/*! \brief Factory function for variable_profile_table
 */
menoh_error_code MENOH_API menoh_build_variable_profile_table(
  const menoh_variable_profile_table_builder_handle builder,
  const menoh_model_data_handle model_data,
  menoh_variable_profile_table_handle* dst_handle);
/*! \brief Delete function for variable_profile_table
 *
 * Users must call to release memory resources allocated for
 * variable_profile_table.
 */
void MENOH_API menoh_delete_variable_profile_table(
  menoh_variable_profile_table_handle variable_profile_table);

/*! \brief Accessor function for variable_profile_table
 *
 * Select variable name and get its dtype.
 */
menoh_error_code MENOH_API menoh_variable_profile_table_get_dtype(
  const menoh_variable_profile_table_handle variable_profile_table,
  const char* variable_name, menoh_dtype* dst_dtype);

/*! \brief Accessor function for variable_profile_table
 *
 * Select variable name and get its dims.size() . (eg if dims of variable "foo"
 * is (32, 128), dims.size() is 2)
 *
 */
menoh_error_code MENOH_API menoh_variable_profile_table_get_dims_size(
  const menoh_variable_profile_table_handle variable_profile_table,
  const char* variable_name, int32_t* dst_size);

/*! \brief Accessor function for variable_profile_table
 *
 * Select variable name and index, then get its dims.at(index). (eg if dims of
 * variable "foo" and index is 1, (32, 128), dims.at(1) is 128)
 *
 */
menoh_error_code MENOH_API menoh_variable_profile_table_get_dims_at(
  const menoh_variable_profile_table_handle variable_profile_table,
  const char* variable_name, int32_t index, int32_t* dst_size);

/** @} */

/*! @ingroup model_data
 * \brief Optimize function for menoh_model_data
 *
 * \note This function modify given model_data.
 */
menoh_error_code MENOH_API menoh_model_data_optimize(
  menoh_model_data_handle model_data,
  const menoh_variable_profile_table_handle variable_profile_table);

/** @addtogroup model Model types and operations
 * @{ */
/*! \struct menoh_model_builder
 *  \brief menoh_model_builder is helper for creation of model.
 *
 * Users can attach external buffers to variables.
 *
 * See menoh_model_builder_attach_external_buffer()
 */
struct menoh_model_builder;
typedef struct menoh_model_builder* menoh_model_builder_handle;

/*! \brief Factory function for menoh_model_builder
 */
menoh_error_code MENOH_API menoh_make_model_builder(
  const menoh_variable_profile_table_handle variable_profile_table,
  menoh_model_builder_handle* dst_handle);
/*! \brief Delete function for model_builder
 *
 * Users must call to release memory resources allocated for model_builder
 */
void MENOH_API
menoh_delete_model_builder(menoh_model_builder_handle model_builder);

/*! \brief Attach a buffer which allocated by users.
 *
 * Users can attach a external buffer which they allocated to target variable.
 *
 * Variables attached no external buffer are attached internal buffers allocated
 * automatically.
 *
 * \note Users can get that internal buffer handle by calling
 * menoh_model_get_variable_buffer_handle() later.
 */
menoh_error_code MENOH_API menoh_model_builder_attach_external_buffer(
  menoh_model_builder_handle builder, const char* variable_name,
  void* buffer_handle);

/*! \struct menoh_model
 *  \brief menoh_model is the main component to execute model inference.
 *
 *  See menoh_model_run()
 */
struct menoh_model;
typedef struct menoh_model* menoh_model_handle;

/*! \brief Factory function for menoh_model
 *
 * \note Users can (and should) delete model_data after the model creation by
 * calling menoh_delete_model_data().
 */
menoh_error_code MENOH_API menoh_build_model(
  const menoh_model_builder_handle builder,
  const menoh_model_data_handle model_data, const char* backend_name,
  const char* backend_config, menoh_model_handle* dst_model_handle);
/*! \brief Delete function for model
 *
 * Users must call to release memory resources allocated for model
 */
void MENOH_API menoh_delete_model(menoh_model_handle model);

/*! \brief Get a buffer handle attached to target variable.
 *
 * Users can get a buffer handle attached to target variable.
 *
 * If that buffer is allocated by users and attached to the variable by calling
 * menoh_model_builder_attach_external_buffer(), returned buffer handle is same
 * to it.
 *
 * \note Automatically allocated internal buffers are released automatically so
 * users do not need to (and must not) release them.
 */
menoh_error_code MENOH_API menoh_model_get_variable_buffer_handle(
  const menoh_model_handle model, const char* variable_name, void** dst_data);

/*! \brief Get dtype of target variable.
 *
 */
menoh_error_code MENOH_API menoh_model_get_variable_dtype(
  const menoh_model_handle model, const char* variable_name,
  menoh_dtype* dst_dtype);

/*! \brief Get size of dims of target variable.
 *
 * \sa
 * menoh_variable_profile_table_get_dims_size()
 */
menoh_error_code MENOH_API menoh_model_get_variable_dims_size(
  const menoh_model_handle model, const char* variable_name, int32_t* dst_size);

/*! \brief Get an element of dims of target variable specified by index.
 *
 * \sa
 * menoh_variable_profile_table_get_dims_at()
 */
menoh_error_code MENOH_API menoh_model_get_variable_dims_at(
  const menoh_model_handle model, const char* variable_name, int32_t index,
  int32_t* dst_size);

/*! \brief Run model inference
 *
 * \warning This function can't be called asynchronously.
 */
menoh_error_code MENOH_API menoh_model_run(menoh_model_handle model);

/** @} */

/** @} */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif // MENOH_C_INTERFACE_H
