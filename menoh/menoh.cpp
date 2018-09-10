#include <algorithm>
#include <array>
#include <iterator>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <menoh/menoh.h>

#include <menoh/exception.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_core_factory.hpp>
#include <menoh/model_data.hpp>
#include <menoh/onnx.hpp>
#include <menoh/utility.hpp>

namespace menoh_impl {
    using fixed_array = std::array<char, MENOH_ERROR_MESSAGE_MAX_LENGTH>;
    fixed_array& get_error_message_singleton() noexcept {
        thread_local fixed_array message = {'\0'};
        return message;
    }

    void set_last_error_message(const char* message) noexcept {
        auto& arr = get_error_message_singleton();
        auto message_size =
          std::char_traits<char>::length(message) + 1; // +1 for null char
        if(arr.size() < message_size) {
            const char* prefix =
              "An error occured, and its log message is longer than prepared. "
              "To view all message, please extend "
              "\"menoh_error_message_max_length\" (all capitals) macro: ";
            auto cont =
              std::copy(prefix, prefix + std::char_traits<char>::length(prefix),
                        arr.begin());
            std::copy(message,
                      message + (static_cast<size_t>(arr.end() - cont) - 1),
                      cont);

        } else {
            std::copy(message, message + message_size, arr.data());
        }
    }
} // namespace menoh_impl

#undef MENOH_ERROR_MESSAGE_MAX_LENGTH

const char* menoh_get_last_error_message() {
    return menoh_impl::get_error_message_singleton().data();
}

template <typename Func>
menoh_error_code check_error(Func func) {
    try {
        menoh_error_code ec = func();
        if(ec) {
            return ec;
        }
    } catch(menoh_impl::exception const& e) {
        menoh_impl::set_last_error_message(e.what());
        return e.error_code(); //
    } catch(std::exception const& e) {
        menoh_impl::set_last_error_message(e.what());
        return menoh_error_code_std_error; //
    } catch(...) {
        menoh_impl::set_last_error_message("");
        return menoh_error_code_unknown_error; //
    }
    return menoh_error_code_success;
}

/*
 * model_data
 */
struct menoh_model_data {
    menoh_impl::model_data model_data;
};

menoh_error_code menoh_make_model_data(menoh_model_data_handle* dst_handle) {
    return check_error([&]() {
        *dst_handle = std::make_unique<menoh_model_data>().release();
        return menoh_error_code_success;
    });
}

menoh_error_code MENOH_API menoh_model_data_add_new_node(
  menoh_model_data* model_data, const char* op_type) {
    return check_error([&]() {
        model_data->model_data.node_list.push_back({op_type, {}, {}, {}});
        return menoh_error_code_success;
    });
}

menoh_error_code MENOH_API menoh_model_data_add_input_name_to_current_node(
  menoh_model_data* model_data, const char* input_name) {
    return check_error([&]() {
        model_data->model_data.node_list.back().input_name_list.push_back(
          input_name);
        return menoh_error_code_success;
    });
}

menoh_error_code MENOH_API menoh_model_data_add_output_name_to_current_node(
  menoh_model_data* model_data, const char* output_name) {
    return check_error([&]() {
        model_data->model_data.node_list.back().output_name_list.push_back(
          output_name);
        return menoh_error_code_success;
    });
}

menoh_error_code MENOH_API menoh_model_data_add_attribute_int_to_current_node(
  menoh_model_data_handle model_data, const char* attribute_name,
  int32_t value) {
    return check_error([&]() {
        model_data->model_data.node_list.back().attribute_table.insert(
          {std::string(attribute_name), value});
        return menoh_error_code_success;
    });
}

menoh_error_code MENOH_API menoh_model_data_add_attribute_float_to_current_node(
  menoh_model_data_handle model_data, const char* attribute_name, float value) {
    return check_error([&]() {
        model_data->model_data.node_list.back().attribute_table.insert(
          {std::string(attribute_name), value});
        return menoh_error_code_success;
    });
}

menoh_error_code MENOH_API menoh_model_data_add_attribute_ints_to_current_node(
  menoh_model_data_handle model_data, const char* attribute_name, int32_t size,
  const int* value) {
    return check_error([&]() {
        model_data->model_data.node_list.back().attribute_table.insert(
          {std::string(attribute_name),
           std::vector<int32_t>(value, value + size)});
        return menoh_error_code_success;
    });
}

menoh_error_code MENOH_API
menoh_model_data_add_attribute_floats_to_current_node(
  menoh_model_data_handle model_data, const char* attribute_name, int32_t size,
  const float* value) {
    return check_error([&]() {
        model_data->model_data.node_list.back().attribute_table.insert(
          {std::string(attribute_name),
           std::vector<float>(value, value + size)});
        return menoh_error_code_success;
    });
}

menoh_error_code MENOH_API menoh_model_data_add_initializer(
  menoh_model_data* model_data, const char* initializer_name, menoh_dtype dtype,
  int32_t dims_size, const int32_t* dims, void* buffer_handle) {
    return check_error([&]() {
        model_data->model_data.parameter_name_and_array_list.push_back(
          {std::string(initializer_name),
           menoh_impl::array(static_cast<menoh_impl::dtype_t>(dtype),
                             std::vector<int32_t>(dims, dims + dims_size),
                             buffer_handle)});
        return menoh_error_code_success;
    });
}

// menoh_error_code menoh_model_data_add_node(const char* op_type, )

menoh_error_code
menoh_make_model_data_from_onnx(const char* onnx_filename,
                                menoh_model_data_handle* dst_handle) {
    return check_error([&]() {
        *dst_handle =
          std::make_unique<menoh_model_data>(
            menoh_model_data{
              menoh_impl::make_model_data_from_onnx_file(onnx_filename)})
            .release();
        return menoh_error_code_success;
    });
}
menoh_error_code menoh_make_model_data_from_onnx_data_on_memory(
  const uint8_t* onnx_data, int32_t size, menoh_model_data_handle* dst_handle) {
    return check_error([&]() {
        *dst_handle = std::make_unique<menoh_model_data>(
                        menoh_model_data{
                          menoh_impl::make_model_data_from_onnx_data_on_memory(
                            onnx_data, size)})
                        .release();
        return menoh_error_code_success;
    });
}
void menoh_delete_model_data(menoh_model_data_handle model_data) {
    delete model_data;
}

/*
 * variable_profile_table_builder
 */
struct menoh_variable_profile_table_builder {
    std::vector<std::tuple<std::string, menoh_dtype, std::vector<int>>>
      input_name_and_dtype_and_dims_list;
    std::vector<std::tuple<std::string, menoh_dtype>>
      output_name_and_dtype_list;
};

menoh_error_code menoh_make_variable_profile_table_builder(
  menoh_variable_profile_table_builder_handle* dst_handle) {
    return check_error([&]() {
        *dst_handle =
          std::make_unique<menoh_variable_profile_table_builder>().release();
        return menoh_error_code_success;
    });
}
void menoh_delete_variable_profile_table_builder(
  menoh_variable_profile_table_builder_handle builder) {
    delete builder;
}

menoh_error_code menoh_variable_profile_table_builder_add_input_profile(
  menoh_variable_profile_table_builder_handle builder, const char* name,
  menoh_dtype dtype, int32_t dims_size, const int32_t* dims) {
    return check_error([&]() {
        builder->input_name_and_dtype_and_dims_list.push_back(
          std::make_tuple(std::string(name), dtype,
                          std::vector<int32_t>(dims, dims + dims_size)));
        return menoh_error_code_success;
    });
}

menoh_error_code menoh_variable_profile_table_builder_add_input_profile_dims_2(
  menoh_variable_profile_table_builder_handle builder, const char* name,
  menoh_dtype dtype, int32_t num, int32_t size) {
    return check_error([&]() {
        std::vector<int> dims = {num, size};
        builder->input_name_and_dtype_and_dims_list.push_back(
          std::make_tuple(std::string(name), dtype, dims));
        return menoh_error_code_success;
    });
}
menoh_error_code menoh_variable_profile_table_builder_add_input_profile_dims_4(
  menoh_variable_profile_table_builder_handle builder, const char* name,
  menoh_dtype dtype, int32_t num, int32_t channel, int32_t height,
  int32_t width) {
    return check_error([&]() {
        std::vector<int> dims = {num, channel, height, width};
        builder->input_name_and_dtype_and_dims_list.push_back(
          std::make_tuple(std::string(name), dtype, dims));
        return menoh_error_code_success;
    });
}

menoh_error_code menoh_variable_profile_table_builder_add_output_profile(
  menoh_variable_profile_table_builder_handle builder, const char* name,
  menoh_dtype dtype) {
    return check_error([&]() {
        auto found = std::find_if(
          builder->output_name_and_dtype_list.begin(),
          builder->output_name_and_dtype_list.end(),
          [name](auto const& t) { return name == std::get<0>(t); });
        if(found != builder->output_name_and_dtype_list.end()) {
            auto message =
              std::string("menoh same named variable already exist: ") + name;
            menoh_impl::set_last_error_message(message.c_str());
            return menoh_error_code_same_named_variable_already_exist;
        }
        builder->output_name_and_dtype_list.push_back(
          std::make_tuple(std::string(name), dtype));
        return menoh_error_code_success;
    });
}

/*
 * variable_profile_table
 */
struct menoh_variable_profile_table {
    std::unordered_map<std::string, std::tuple<menoh_dtype, std::vector<int>>>
      input_profile_table;
    std::unordered_map<std::string, std::tuple<menoh_dtype, std::vector<int>>>
      output_profile_table;
};

menoh_error_code menoh_build_variable_profile_table(
  const menoh_variable_profile_table_builder_handle builder,
  const menoh_model_data_handle model_data,
  menoh_variable_profile_table_handle* dst_handle) {
    return check_error([&]() {
        std::unordered_map<std::string,
                           std::tuple<menoh_dtype, std::vector<int>>>
          input_profile_table;
        std::transform(
          builder->input_name_and_dtype_and_dims_list.begin(),
          builder->input_name_and_dtype_and_dims_list.end(),
          std::inserter(input_profile_table, input_profile_table.end()),
          [](auto const& t) {
              return std::make_pair(
                std::get<0>(t),
                std::make_tuple(std::get<1>(t), std::get<2>(t)));
          });

        std::vector<std::pair<std::string, std::vector<int>>>
          input_name_and_dims_pair_list;
        std::transform(
          builder->input_name_and_dtype_and_dims_list.begin(),
          builder->input_name_and_dtype_and_dims_list.end(),
          std::back_inserter(input_name_and_dims_pair_list), [](auto const& t) {
              return std::make_pair(std::get<0>(t), std::get<2>(t));
          });
        auto output_dims_table = menoh_impl::make_output_dims_table(
          model_data->model_data, input_name_and_dims_pair_list);

        std::unordered_map<std::string,
                           std::tuple<menoh_dtype, std::vector<int>>>
          output_profile_table;
        std::transform(
          builder->output_name_and_dtype_list.begin(),
          builder->output_name_and_dtype_list.end(),
          std::inserter(output_profile_table, output_profile_table.end()),
          [&output_dims_table](auto const& t) {
              std::string name;
              menoh_dtype dtype;
              std::tie(name, dtype) = t;
              return std::make_pair(
                name, std::make_tuple(dtype, menoh_impl::find_value(
                                               output_dims_table, name)));
          });
        *dst_handle =
          std::make_unique<menoh_variable_profile_table>(
            menoh_variable_profile_table{std::move(input_profile_table),
                                         std::move(output_profile_table)})
            .release();
        return menoh_error_code_success;
    });
}
void menoh_delete_variable_profile_table(
  menoh_variable_profile_table_handle variable_profile_table) {
    delete variable_profile_table;
}

namespace impl {
    template <typename F>
    menoh_error_code menoh_variable_profile_table_get_variable_attribute(
      const menoh_variable_profile_table_handle variable_profile_table,
      const char* name, F f) {
        return check_error([&]() {
            auto outiter =
              variable_profile_table->output_profile_table.find(name);
            if(outiter != variable_profile_table->output_profile_table.end()) {
                f(outiter->second);
            } else {
                auto initer =
                  variable_profile_table->input_profile_table.find(name);
                if(initer !=
                   variable_profile_table->input_profile_table.end()) {
                    f(initer->second);
                } else {
                    auto message =
                      std::string("menoh variable not found: ") + name;
                    menoh_impl::set_last_error_message(message.c_str());
                    return menoh_error_code_variable_not_found;
                }
            }
            return menoh_error_code_success;
        });
    }
} // namespace impl

menoh_error_code menoh_variable_profile_table_get_dtype(
  const menoh_variable_profile_table_handle variable_profile_table,
  const char* name, menoh_dtype* dst_dtype) {
    return impl::menoh_variable_profile_table_get_variable_attribute(
      variable_profile_table, name,
      [&](std::tuple<menoh_dtype, std::vector<int>> const& t) {
          *dst_dtype = std::get<0>(t);
      });
}
menoh_error_code menoh_variable_profile_table_get_dims_size(
  const menoh_variable_profile_table_handle variable_profile_table,
  const char* name, int32_t* dst_size) {
    return impl::menoh_variable_profile_table_get_variable_attribute(
      variable_profile_table, name,
      [&](std::tuple<menoh_dtype, std::vector<int>> const& t) {
          *dst_size = static_cast<int32_t>(std::get<1>(t).size());
      });
}
menoh_error_code menoh_variable_profile_table_get_dims_at(
  const menoh_variable_profile_table_handle variable_profile_table,
  const char* name, int32_t index, int32_t* dst_size) {
    return impl::menoh_variable_profile_table_get_variable_attribute(
      variable_profile_table, name,
      [&](std::tuple<menoh_dtype, std::vector<int>> const& t) {
          *dst_size = std::get<1>(t).at(index);
      });
}

menoh_error_code menoh_model_data_optimize(
  menoh_model_data_handle model_data,
  const menoh_variable_profile_table_handle variable_profile_table) {
    return check_error([&]() {
        std::vector<std::string> required_output_name_list;
        std::transform(variable_profile_table->output_profile_table.begin(),
                       variable_profile_table->output_profile_table.end(),
                       std::back_inserter(required_output_name_list),
                       [](auto const& e) { return e.first; });
        auto optimized = trim_redundant_nodes(model_data->model_data,
                                              required_output_name_list);
        std::swap(model_data->model_data, optimized);
        return menoh_error_code_success;
    });
}

/*
 * model builder
 */
struct menoh_model_builder {
    std::unordered_map<std::string, std::tuple<menoh_dtype, std::vector<int>>>
      input_profile_table;
    std::unordered_map<std::string, std::tuple<menoh_dtype, std::vector<int>>>
      output_profile_table;
    std::unordered_map<std::string, void*> external_buffer_handle_table;
};

menoh_error_code menoh_make_model_builder(
  const menoh_variable_profile_table_handle variable_profile_table,
  menoh_model_builder_handle* dst_handle) {
    return check_error([&]() {
        *dst_handle =
          std::make_unique<menoh_model_builder>(
            menoh_model_builder{variable_profile_table->input_profile_table,
                                variable_profile_table->output_profile_table,
                                {}})
            .release();
        return menoh_error_code_success;
    });
}
void menoh_delete_model_builder(menoh_model_builder_handle builder) {
    delete builder;
}

menoh_error_code menoh_model_builder_attach_external_buffer(
  menoh_model_builder_handle builder, const char* name, void* buffer_handle) {
    return check_error([&]() {
        auto found =
          std::find_if(builder->external_buffer_handle_table.begin(),
                       builder->external_buffer_handle_table.end(),
                       [name](auto const& p) { return name == p.first; });
        if(found != builder->external_buffer_handle_table.end()) {
            auto message =
              std::string("menoh same named variable already exist: ") + name;
            menoh_impl::set_last_error_message(message.c_str());
            return menoh_error_code_same_named_variable_already_exist;
        }
        builder->external_buffer_handle_table.insert(
          {std::string(name), buffer_handle});
        return menoh_error_code_success;
    });
}

/*
 * model
 */
struct menoh_model {
    std::unordered_map<std::string, menoh_impl::array> input_table;
    std::unordered_map<std::string, menoh_impl::array> output_table;
    std::unique_ptr<menoh_impl::model_core> model_core;
};

/* You can (and should) delete model_data after the model creation. */
menoh_error_code menoh_build_model(const menoh_model_builder_handle builder,
                                   const menoh_model_data_handle model_data,
                                   const char* backend_name,
                                   const char* backend_config,
                                   menoh_model_handle* dst_model_handle) {
    return check_error([&]() {
        std::unordered_map<std::string, menoh_impl::array> input_table;
        for(auto p : builder->input_profile_table) {
            std::string name;
            std::tuple<menoh_dtype, std::vector<int>> t;
            std::tie(name, t) = p;
            menoh_dtype dtype;
            std::vector<int> dims;
            std::tie(dtype, dims) = t;

            auto buff = builder->external_buffer_handle_table.find(name);

            if(buff == builder->external_buffer_handle_table.end()) {
                input_table.insert(
                  {name, menoh_impl::array(
                           static_cast<menoh_impl::dtype_t>(dtype), dims)});
            } else {
                input_table.insert(
                  {name,
                   menoh_impl::array(static_cast<menoh_impl::dtype_t>(dtype),
                                     dims, buff->second)});
            }
        }

        std::unordered_map<std::string, menoh_impl::array> output_table;
        for(auto p : builder->output_profile_table) {
            std::string name;
            std::tuple<menoh_dtype, std::vector<int>> t;
            std::tie(name, t) = p;
            menoh_dtype dtype;
            std::vector<int> dims;
            std::tie(dtype, dims) = t;

            auto buff = builder->external_buffer_handle_table.find(name);

            if(buff == builder->external_buffer_handle_table.end()) {
                output_table.insert(
                  {name, menoh_impl::array(
                           static_cast<menoh_impl::dtype_t>(dtype), dims)});
            } else {
                output_table.insert(
                  {name,
                   menoh_impl::array(static_cast<menoh_impl::dtype_t>(dtype),
                                     dims, buff->second)});
            }
        }

        *dst_model_handle =
          std::make_unique<menoh_model>(
            menoh_model{input_table, output_table,
                        menoh_impl::make_model_core(
                          input_table, output_table, model_data->model_data,
                          backend_name, backend_config)})
            .release();
        return menoh_error_code_success;
    });
}
void menoh_delete_model(menoh_model_handle model) {
    delete model;
}

namespace impl {
    template <typename F>
    menoh_error_code
    menoh_model_get_variable_variable_attribute(const menoh_model_handle model,
                                                const char* name, F f) {
        return check_error([&]() {
            auto outiter = model->output_table.find(name);
            if(outiter != model->output_table.end()) {
                f(outiter->second);
            } else {
                auto initer = model->input_table.find(name);
                if(initer != model->input_table.end()) {
                    f(initer->second);
                } else {
                    auto message =
                      std::string("menoh variable not found: ") + name;
                    menoh_impl::set_last_error_message(message.c_str());
                    return menoh_error_code_variable_not_found;
                }
            }
            return menoh_error_code_success;
        });
    }
} // namespace impl

menoh_error_code menoh_model_get_variable_buffer_handle(
  const menoh_model_handle model, const char* variable_name, void** data_p) {
    return impl::menoh_model_get_variable_variable_attribute(
      model, variable_name,
      [&](menoh_impl::array const& arr) { *data_p = arr.data(); });
}

menoh_error_code menoh_model_get_variable_dtype(const menoh_model_handle model,
                                                const char* variable_name,
                                                menoh_dtype* dst_dtype) {
    return impl::menoh_model_get_variable_variable_attribute(
      model, variable_name, [&](menoh_impl::array const& arr) {
          *dst_dtype = static_cast<menoh_dtype>(arr.dtype());
      });
}

menoh_error_code
menoh_model_get_variable_dims_size(const menoh_model_handle model,
                                   const char* variable_name,
                                   int32_t* dst_size) {
    return impl::menoh_model_get_variable_variable_attribute(
      model, variable_name, [&](menoh_impl::array const& arr) {
          *dst_size = static_cast<int32_t>(arr.dims().size());
      });
}

menoh_error_code
menoh_model_get_variable_dims_at(const menoh_model_handle model,
                                 const char* variable_name, int32_t index,
                                 int32_t* dst_size) {
    return impl::menoh_model_get_variable_variable_attribute(
      model, variable_name,
      [&](menoh_impl::array const& arr) { *dst_size = arr.dims().at(index); });
}

menoh_error_code menoh_model_run(menoh_model_handle model) {
    return check_error([&]() {
        model->model_core->run();
        return menoh_error_code_success;
    });
}
