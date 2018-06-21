#ifndef MENOH_HPP
#define MENOH_HPP

#include <memory>
#include <vector>

#include <menoh/menoh.h>

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#define MENOH_CPP_API_ERROR_CHECK(statement)                 \
    {                                                        \
        auto ec = statement;                                 \
        if(ec) {                                             \
            throw error(ec, menoh_get_last_error_message()); \
        }                                                    \
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS

namespace menoh {
    /** @addtogroup cpp_api C++ API
     * @{ */

    /** @addtogroup cpp_error_handling Error
     * @{ */
    enum class error_code_t {
        success = menoh_error_code_success,
        std_error = menoh_error_code_std_error,
        unknown_error = menoh_error_code_unknown_error,
        invalid_filename = menoh_error_code_invalid_filename,
        unsupported_onnx_opset_version =
          menoh_error_code_unsupported_onnx_opset_version,
        onnx_parse_error = menoh_error_code_onnx_parse_error,
        invalid_dtype = menoh_error_code_invalid_dtype,
        invalid_attribute_type = menoh_error_code_invalid_attribute_type,
        unsupported_operator_attribute =
          menoh_error_code_unsupported_operator_attribute,
        dimension_mismatch = menoh_error_code_dimension_mismatch,
        variable_not_found = menoh_error_code_variable_not_found,
        index_out_of_range = menoh_error_code_index_out_of_range,
        json_parse_error = menoh_error_code_json_parse_error,
        invalid_backend_name = menoh_error_code_invalid_backend_name,
        unsupported_operator = menoh_error_code_unsupported_operator,
        failed_to_configure_operator =
          menoh_error_code_failed_to_configure_operator,
        backend_error = menoh_error_code_backend_error,
        same_named_variable_already_exist =
          menoh_error_code_same_named_variable_already_exist,
        invalid_dims_size,
    };

    //! The error class thrown when any error occured.
    /*!
     * \note
     * Users can get error message by calling what() member function.
     */
    class error : public std::runtime_error {
    public:
        explicit error(menoh_error_code ec, std::string const& message)
          : runtime_error(message), ec_(static_cast<error_code_t>(ec)) {}

        explicit error(error_code_t ec, std::string const& message)
          : error(static_cast<menoh_error_code>(ec), message) {}

        error_code_t error_code() const noexcept { return ec_; }

    private:
        error_code_t ec_;
    };
    /** @} */

    class variable_profile_table;

    /** @addtogroup cpp_model_data Model data
     * @{ */
    //! model data class
    class model_data {
    public:
        /*! \note Normally users needn't call this constructer. Use
         * make_model_data_from_onnx() instead.
         */
        model_data(menoh_model_data_handle h)
          : impl_(h, menoh_delete_model_data) {}

        /*! Accessor to internal handle
         *
         * \note Normally users needn't call this function.
         */
        menoh_model_data_handle get() const noexcept { return impl_.get(); }

        //! Release internal alocated memory.
        /*! Users can (and should) call this function after
         * model_builder::build_model() function call.
         *
         * \warn Do not reuse model_data being reset by this function.
         *
         * \sa
         * model_builder::build_model()
         */
        void reset() noexcept { impl_.reset(); }

        //! Optimize model_data.
        /*! This function modify internal state of model_data.
         */
        void optimize(variable_profile_table const& vpt);

    private:
        std::unique_ptr<menoh_model_data, decltype(&menoh_delete_model_data)>
          impl_;
    };

    //! Load ONNX file and make model_data
    inline model_data
    make_model_data_from_onnx(std::string const& onnx_filename) {
        menoh_model_data_handle h;
        MENOH_CPP_API_ERROR_CHECK(
          menoh_make_model_data_from_onnx(onnx_filename.c_str(), &h));
        return model_data(h);
    }
    /** @} */

    /** @addtogroup cpp_vpt Veriable profile table
     * @{ */
    enum class dtype_t { float_ = menoh_dtype_float };

    struct variable_profile {
        dtype_t dtype;
        std::vector<int32_t> dims;
    };

    //! Key value store for variable_profile
    class variable_profile_table {
    public:
        /*! \note Normally users needn't call this constructer. Use
         * variable_profile_table_builder::build_variable_profile_table()
         * instead.
         */
        explicit variable_profile_table(menoh_variable_profile_table_handle h)
          : impl_(h, menoh_delete_variable_profile_table) {}

        //! Accessor to variable profile.
        /*!
         * \sa variable_profile
         */
        variable_profile get_variable_profile(std::string const& name) const {
            menoh_dtype dtype;
            MENOH_CPP_API_ERROR_CHECK(menoh_variable_profile_table_get_dtype(
              impl_.get(), name.c_str(), &dtype));
            int32_t dims_size;
            MENOH_CPP_API_ERROR_CHECK(
              menoh_variable_profile_table_get_dims_size(
                impl_.get(), name.c_str(), &dims_size));
            std::vector<int32_t> dims(dims_size);
            for(int i = 0; i < dims_size; ++i) {
                MENOH_CPP_API_ERROR_CHECK(
                  menoh_variable_profile_table_get_dims_at(
                    impl_.get(), name.c_str(), i, &dims.at(i)));
            }
            return variable_profile{static_cast<dtype_t>(dtype), dims};
        }

        //! Accessor to internal handle
        /*!
         * \note Normally users needn't call this function.
         */
        menoh_variable_profile_table_handle get() const noexcept {
            return impl_.get();
        }

    private:
        std::unique_ptr<menoh_variable_profile_table,
                        decltype(&menoh_delete_variable_profile_table)>
          impl_;
    };
    /** @} */

    /** @addtogroup cpp_model
     * @{ */
    inline void model_data::optimize(variable_profile_table const& vpt) {
        MENOH_CPP_API_ERROR_CHECK(
          menoh_model_data_optimize(impl_.get(), vpt.get()));
    }
    /** @} */

    /** @addtogroup cpp_vpt Veriable profile table
     * @{ */
    //! The builder class to build variable_profile_table
    class variable_profile_table_builder {
    public:
        variable_profile_table_builder()
          : impl_(nullptr, menoh_delete_variable_profile_table_builder) {
            menoh_variable_profile_table_builder_handle h;
            MENOH_CPP_API_ERROR_CHECK(
              menoh_make_variable_profile_table_builder(&h));
            impl_.reset(h);
        }

        //! Add input profile. That profile contains name, dtype and dims.
        void add_input_profile(std::string const& name, dtype_t dtype,
                               std::vector<int32_t> const& dims) {
            if(dims.size() == 2) {
                MENOH_CPP_API_ERROR_CHECK(
                  menoh_variable_profile_table_builder_add_input_profile_dims_2(
                    impl_.get(), name.c_str(), static_cast<menoh_dtype>(dtype),
                    dims.at(0), dims.at(1)));
            } else if(dims.size() == 4) {
                MENOH_CPP_API_ERROR_CHECK(
                  menoh_variable_profile_table_builder_add_input_profile_dims_4(
                    impl_.get(), name.c_str(), static_cast<menoh_dtype>(dtype),
                    dims.at(0), dims.at(1), dims.at(2), dims.at(3)));
            } else {
                throw error(error_code_t::invalid_dims_size,
                            "menoh invalid dims size error (2 or 4 is valid): "
                            "dims size of " +
                              name + " is specified " +
                              std::to_string(dims.size()));
            }
        }

        //! Add output profile. That profile contains name, dtype.
        /*! dims of output is calculated automatically.
         */
        void add_output_profile(std::string const& name, dtype_t dtype) {
            MENOH_CPP_API_ERROR_CHECK(
              menoh_variable_profile_table_builder_add_output_profile(
                impl_.get(), name.c_str(), static_cast<menoh_dtype>(dtype)));
        }

        //! Factory function for variable_profile_table.
        variable_profile_table
        build_variable_profile_table(model_data const& model_data) {
            menoh_variable_profile_table_handle h;
            MENOH_CPP_API_ERROR_CHECK(menoh_build_variable_profile_table(
              impl_.get(), model_data.get(), &h));
            return variable_profile_table(h);
        }

    private:
        std::unique_ptr<menoh_variable_profile_table_builder,
                        decltype(&menoh_delete_variable_profile_table_builder)>
          impl_;
    };
    /** @} */

    /*! @addtogroup cpp_model Model
     * @{ */
    struct variable {
        dtype_t dtype;
        std::vector<int32_t> dims;
        void* buffer_handle;
    };

    //! The main component to run inference.
    class model {
    public:
        /*! \note Normally users needn't call this constructer. Use
         * model_builder::build_model() instead.
         */
        explicit model(menoh_model_handle h) : impl_(h, menoh_delete_model) {}

        //! Accsessor to internal variable.
        /*!
         * \sa
         * variable
         */
        variable get_variable(std::string const& name) const {
            void* buff;
            MENOH_CPP_API_ERROR_CHECK(menoh_model_get_variable_buffer_handle(
              impl_.get(), name.c_str(), &buff));
            menoh_dtype dtype;
            MENOH_CPP_API_ERROR_CHECK(menoh_model_get_variable_dtype(
              impl_.get(), name.c_str(), &dtype));
            int32_t dims_size;
            MENOH_CPP_API_ERROR_CHECK(menoh_model_get_variable_dims_size(
              impl_.get(), name.c_str(), &dims_size));
            std::vector<int32_t> dims(dims_size);
            for(int i = 0; i < dims_size; ++i) {
                MENOH_CPP_API_ERROR_CHECK(menoh_model_get_variable_dims_at(
                  impl_.get(), name.c_str(), i, &dims.at(i)));
            }
            return variable{static_cast<dtype_t>(dtype), dims, buff};
        }

        //! Run model inference.
        /*!
         * \warning
         * Do not call this function asynchronously.
         */
        void run() { menoh_model_run(impl_.get()); }

    private:
        std::unique_ptr<menoh_model, decltype(&menoh_delete_model)> impl_;
    };

    /*! \brief The builder class to build model.
     *
     */
    class model_builder {
    public:
        explicit model_builder(
          variable_profile_table const& variable_profile_table)
          : impl_(nullptr, menoh_delete_model_builder) {
            menoh_model_builder_handle h;
            MENOH_CPP_API_ERROR_CHECK(
              menoh_make_model_builder(variable_profile_table.get(), &h));
            impl_.reset(h);
        }

        //! Users can attach external buffers to variables.
        /*! Variables attached no external buffer are attached internal buffers
         * allocated automatically.
         *
         * \note Users can get that internal buffer handle by calling
         * model::get_variable() later.
         */
        void attach_external_buffer(std::string const& name,
                                    void* buffer_handle) {
            MENOH_CPP_API_ERROR_CHECK(
              menoh_model_builder_attach_external_buffer(
                impl_.get(), name.c_str(), buffer_handle));
        }

        //! Factory function for model
        /*! \note
         * Current supported backend_name is only "mkldnn" and don't set backend
         * config.
         */
        model build_model(model_data const& model_data,
                          std::string const& backend_name,
                          std::string const& backend_config = "") {
            menoh_model_handle h;
            MENOH_CPP_API_ERROR_CHECK(menoh_build_model(
              impl_.get(), model_data.get(), backend_name.c_str(),
              backend_config.c_str(), &h));
            return model(h);
        }

    private:
        std::unique_ptr<menoh_model_builder,
                        decltype(&menoh_delete_model_builder)>
          impl_;
    };
    /** @} */

    /** @} */

} // namespace menoh

#undef MENOH_CPP_API_ERROR_CHECK
#endif // MENOH_HPP
