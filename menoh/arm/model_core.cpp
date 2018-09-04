
#include <menoh/json.hpp>

#include <menoh/arm/model_core.hpp>

using namespace armnn;

namespace menoh_impl {
    namespace armnn_backend {

        model_core::model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
	  menoh_impl::model_data const& model_data,
          std::vector<armnn::Compute>& ComputeDevice,
	  bool EnableFp16TurboMode )
          : m_inference(Params(&input_table, &output_table, &model_data,
			       ComputeDevice, EnableFp16TurboMode)) {}

        void model_core::do_run() {
	    m_inference.Run();
        }

        model_core make_model_core(
          std::unordered_map<std::string, array> const& input_table,
          std::unordered_map<std::string, array> const& output_table,
          menoh_impl::model_data const& model_data,
	  backend_config const& config) {
            try {
                int device = 0;
                if(!config.empty()) {
                    auto c = nlohmann::json::parse(config);
                    if(c.find("device") != c.end()) {
		      device = c["device"].get<int>();
                    }
                }

#ifdef ANDROID_ARM
		std::vector<armnn::Compute> compute{};
#ifdef ENABLE_OPENCL
                device = 2;
#else
                device = 1;
#endif
#else
		std::vector<armnn::Compute> compute{armnn::Compute::CpuRef};
		std::cout << "armnn::Compute::CpuRef" << std::endl;
#endif

                if( device == 0 )
		    std::cout << "armnn::Compute::CpuRef" << std::endl;
		else if( device == 1 ) {
		    compute.push_back(armnn::Compute::CpuAcc);
		    std::cout << "armnn::Compute::CpuAcc" << std::endl;
                } else if( device == 2 ) {
		    compute.push_back(armnn::Compute::GpuAcc);
		    std::cout << "armnn::Compute::GpuAcc" << std::endl;
		} else 
                    throw json_parse_error("Device Error");
                if( device != 0 && device != 1 && device != 2 )
                    throw json_parse_error("Device Error");

	        boost::ignore_unused(compute);
	    
                return model_core(input_table, output_table, model_data, compute, false /*true*/);
            } catch(nlohmann::json::parse_error const& e) {
                throw json_parse_error(e.what());
            }
        }

    } // namespace armnn_backend
} // namespace menoh_impl
