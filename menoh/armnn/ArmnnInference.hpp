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

#include <menoh/armnn/MenohParser.hpp>

using namespace armnn;

namespace menoh_impl {

  namespace armnn_backend {

        struct Params
        {
 	    armnn::IRuntime::CreationOptions options; // default options
            std::vector<armnn::Compute> m_ComputeDevice;
            bool m_EnableFp16TurboMode;
	    std::unordered_map<std::string, array> const* input_table_;
 	    std::unordered_map<std::string, array> const* output_table_;
            menoh_impl::model_data const* model_data_;

            Params()
		: options()
	        , m_ComputeDevice{armnn::Compute::CpuRef}
	        , m_EnableFp16TurboMode(false)
	        , input_table_(nullptr)
	        , output_table_(nullptr)
	        , model_data_(nullptr) {}

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
        };

        class ArmnnInference {
        public:

	    ArmnnInference( const Params& param );

            void Run();

        private:

            void Build( graph menoh_graph,
                        std::unordered_map<std::string, array> const& parameter_table,
                        std::map<std::string, TensorShape>& inputShapes,
                        std::vector<std::string>& requestedOutputs );

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
                        // Coverity fix: it should not be possible to get here but boost::str
	                // and boost::format can both
                        // throw uncaught exceptions - convert them to armnn exceptions and rethrow
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
            MenohParser         m_Parser;
	    armnn::IRuntimePtr 	m_Runtime;
	    armnn::NetworkId    m_NetworkIdentifier;

            std::vector<std::string> input_name_list;
            std::vector<std::string> output_name_sorted_list;

            std::unordered_map<std::string, array> m_Input;
	    std::unordered_map<std::string, array> m_Output;
	  
	    std::pair<armnn::LayerBindingId, armnn::TensorInfo> m_InputBindingInfo;
            std::pair<armnn::LayerBindingId, armnn::TensorInfo> m_OutputBindingInfo;
        };

    } // namespace armnn_backend
} // namespace menoh_impl
#endif // ARMNN_INFERENCE_HPP
