#ifndef ARMNN_INFERENCE_HPP
#define ARMNN_INFERENCE_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/log/trivial.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/polymorphic_cast.hpp>
#include <boost/exception/exception.hpp>
#include <boost/exception/diagnostic_information.hpp>

#include <armnn/ArmNN.hpp>

#include <menoh/array.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>

#include <menoh/armnn/Parser.hpp>

using namespace armnn;

namespace menoh_impl {

  namespace armnn_backend {

        struct Params
        {
            Params(
		std::unordered_map<std::string, array> const* input_table,
		std::unordered_map<std::string, array> const* output_table,
                menoh_impl::model_data const* model_data,
                std::vector<armnn::Compute>& ComputeDevice,
		bool EnableFp16TurboMode = false )
		: options()
	        , m_ComputeDevice(ComputeDevice)
	        , m_EnableFp16TurboMode(EnableFp16TurboMode)
	        , input_table_(input_table)
		, output_table_(output_table)
	        , model_data_(model_data) {}

 	    armnn::IRuntime::CreationOptions options; // default options
            std::vector<armnn::Compute> m_ComputeDevice;
            bool m_EnableFp16TurboMode;
	    std::unordered_map<std::string, array> const* input_table_;
 	    std::unordered_map<std::string, array> const* output_table_;
            menoh_impl::model_data const* model_data_;
        };

        class Inference {
        public:

	    Inference( const Params& param );

            void Run();

        private:

            void Build( menoh_impl::graph menoh_graph,
                        std::unordered_map<std::string, array> const& parameter_table,
                        std::vector<std::string>& outputs );

            InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId, TensorInfo>& input,
                                          const array& inputTensorData) {
                unsigned int size = (unsigned int)total_size(inputTensorData);
    	        if (size != input.second.GetNumElements())
                {
                    try
                    {
                        throw armnn::Exception(boost::str(boost::format("Input tensor has incorrect size."
			  					        " Expected %1% elements but got %2%.")
					                                % input.second.GetNumElements() % size));
                    } catch (const boost::exception& e)
                    {
                        throw armnn::Exception(diagnostic_information(e));
                    }
                }
                return { { input.first, armnn::ConstTensor(input.second, inputTensorData.data()) } };
            }

           OutputTensors MakeOutputTensors(const std::pair<armnn::LayerBindingId, TensorInfo>& output,
                                           const array& outputTensorData) {
                unsigned int size = (unsigned int)total_size(outputTensorData);
		if (size != output.second.GetNumElements())
                {
                    throw armnn::Exception("Output tensor has incorrect size");
                }
                outputTensorData.data();
                return { { output.first, armnn::Tensor(output.second, outputTensorData.data()) } };
            }

            InputTensors MakeInputTensors(std::unordered_map<std::string, array> const& inputTensorData) {
	        return MakeInputTensors(m_InputBindingInfo, inputTensorData.begin()->second);
            }

            OutputTensors MakeOutputTensors(std::unordered_map<std::string, array> const& outputTensorData){
	        return MakeOutputTensors(m_OutputBindingInfo, outputTensorData.begin()->second);
            }

            std::vector<armnn::Compute> m_ComputeDevice;
            Parser              m_Parser;
	    armnn::IRuntimePtr 	m_Runtime;
	    armnn::NetworkId    m_NetworkIdentifier;

            std::vector<std::string> input_name_list;

            std::unordered_map<std::string, array> m_Input;
	    std::unordered_map<std::string, array> m_Output;
	  
	    std::pair<armnn::LayerBindingId, armnn::TensorInfo> m_InputBindingInfo;
            std::pair<armnn::LayerBindingId, armnn::TensorInfo> m_OutputBindingInfo;
        };

    } // namespace armnn_backend
} // namespace menoh_impl
#endif // INFERENCE_HPP
