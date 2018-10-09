#include <menoh/model.hpp>

#include <menoh/utility.hpp>

namespace menoh_impl {

    model::model(
      std::vector<
        std::tuple<std::string, dtype_t, std::vector<int>, void*>> const&
        input_name_and_dtype_and_dims_and_data_handle_list,
      std::vector<std::tuple<std::string, dtype_t, void*>> const&
        required_output_name_and_dtype_and_data_handle_list,
      menoh_impl::model_data const& model_data, std::string const& backend_name,
      backend_config const& config) {
        std::vector<std::pair<std::string, std::vector<int>>>
          input_name_and_dims_pair_list;
        for(auto const& t :
            input_name_and_dtype_and_dims_and_data_handle_list) {
            std::string input_name;
            dtype_t dtype;
            std::vector<int> input_dims;
            void* data_handle;
            std::tie(input_name, dtype, input_dims, data_handle) = t;
            assert(input_table_.find(input_name) == input_table_.end());
            if(data_handle) {
                input_table_.insert(
                  {input_name, array(dtype, input_dims, data_handle)});
            } else {
                input_table_.insert({input_name, array(dtype, input_dims)});
            }
            input_name_and_dims_pair_list.push_back({input_name, input_dims});
        }

        auto output_dims_table =
          make_output_dims_table(model_data, input_name_and_dims_pair_list);

        std::unordered_map<std::string, array> output_table;
        for(auto const& t :
            required_output_name_and_dtype_and_data_handle_list) {
            std::string output_name;
            dtype_t dtype;
            void* data_handle;
            std::tie(output_name, dtype, data_handle) = t;
            assert(output_table_.find(output_name) == output_table_.end());
            if(data_handle) {
                output_table_.insert(
                  {output_name,
                   array(dtype, find_value(output_dims_table, output_name),
                         data_handle)});
            } else {
                output_table_.insert(
                  {output_name,
                   array(dtype, find_value(output_dims_table, output_name))});
            }
        }

        model_ = make_model_core(input_table_, output_table_, model_data,
                                 backend_name, config);
    }

    array const& model::input(std::string const& name) const {
        return find_value(input_table_, name);
    }

    array const& model::output(std::string const& name) const {
        return find_value(output_table_, name);
    }

    void model::run() { model_->run(); }

} // namespace menoh_impl
