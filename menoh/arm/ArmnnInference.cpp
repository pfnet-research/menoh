
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <tuple>

#include <menoh/array.hpp>
#include <menoh/exception.hpp>
#include <menoh/graph.hpp>
#include <menoh/json.hpp>
#include <menoh/utility.hpp>

#include <armnnUtils/Permute.hpp>
#include <armnnUtils/GraphTopologicalSort.hpp>

#include <menoh/arm/ArmnnInference.hpp>

using namespace armnn;

namespace menoh_impl {
    namespace armnn_backend {

        ArmnnInference::ArmnnInference( const Params& params )
	  : m_Parser(),
	    m_Runtime(armnn::IRuntime::Create(params.m_ComputeDevice))
	{
	    menoh_impl::model_data const& model_data = *(params.model_data_);

	    std::vector<node> all_nodes;
            std::copy ( model_data.node_list.begin(), model_data.node_list.end(), back_inserter(all_nodes) );

            std::vector<std::string> input_name_list;
            std::map<std::string, armnn::TensorShape> inputShapes;

	    std::unordered_map<std::string, array> const& input_table = *(params.input_table_);
            for(auto const& name_and_arr_pair : input_table) {
                std::string name;
                array arr;
                std::tie(name, arr) = name_and_arr_pair;
                auto dims = arr.dims();
                inputShapes[name] = armnn::TensorShape(dims.size(), (const unsigned int*)dims.data());
                input_name_list.push_back(name);
#ifdef ARM_DEBUG  
	        std::cout << "Input name(" << name << ") : dims(" << arr.dims().size() << ")  = ( ";
		for( auto size : arr.dims() ) std::cerr << size << " "; 
		std::cout << ")" << std::endl;
#endif
                m_Input[name] = arr;
		std::vector<std::string> inputs, outputs;
		outputs.push_back(name);
                std::unordered_map<std::string, attribute> attribute;
                attribute.insert({"dims", std::vector<int>(arr.dims().begin(), arr.dims().end())});
		menoh_impl::node n{"Placeholder", inputs, outputs, attribute};
		all_nodes.push_back(n);
            }	    

            auto parameter_table = std::unordered_map<std::string, array>(
                                        model_data.parameter_name_and_array_list.begin(),
                                        model_data.parameter_name_and_array_list.end());
            for(auto const& param : parameter_table) {
                std::string name;
                array arr;
                std::tie(name, arr) = param;
                auto dims = arr.dims();
                inputShapes[name] = armnn::TensorShape(dims.size(), (const unsigned int*)dims.data());
                input_name_list.push_back(name);
#ifdef ARM_DEBUG
	        std::cout << " Param : " << name << ", dims(" << arr.dims().size() << ")  = ( ";
		for( auto size : arr.dims() ) std::cerr << size << " "; 
		std::cout << ")" << std::endl;
#endif		
		std::vector<std::string> inputs, outputs;
		outputs.push_back(name);
                std::unordered_map<std::string, attribute> attribute;
                attribute.insert({"dims", std::vector<int>(arr.dims().begin(), arr.dims().end())});
		menoh_impl::node n{"Const", inputs, outputs, attribute};
		all_nodes.push_back(n);
            }	    

	    std::unordered_map<std::string, array> const& output_table = *(params.output_table_);
            if (output_table.size() == 0)
            {
                throw ParseException("output must have at least one entry");
            }
 
	    for(auto const& name_and_arr_pair : output_table) {
                std::string name;
                array arr;
                std::tie(name, arr) = name_and_arr_pair;
                auto dims = arr.dims();
#ifdef ARM_DEBUG
	        std::cout << "Output name(" << name << ") : dims(" << arr.dims().size() << ")  = ( ";
		for( auto size : arr.dims() ) std::cerr << size << " "; 
		std::cout << ")" << std::endl;
#endif
                m_Output[name] = arr;
            }	    

            std::vector<std::string> output_name_sorted_list;
            {
                std::transform(output_table.begin(), output_table.end(),
                               std::back_inserter(output_name_sorted_list),
                               [](auto const& e) { return e.first; });
                std::sort(output_name_sorted_list.begin(),
                          output_name_sorted_list.end());
            }

	    std::vector<std::string> requestedOutputs{ output_name_sorted_list };

            auto graph = make_graph(all_nodes);	    
            armnn::INetworkPtr network{nullptr, [](armnn::INetwork *){}};
            {
                auto parameter_table = std::unordered_map<std::string, array>(
                                                         model_data.parameter_name_and_array_list.begin(),
                                                         model_data.parameter_name_and_array_list.end());
 	        network = m_Parser.CreateNetworkFromGraph( graph, parameter_table, inputShapes, requestedOutputs );
            } 

            m_InputBindingInfo  = m_Parser.GetNetworkInputBindingInfo(input_name_list.at(0));
	    m_OutputBindingInfo = m_Parser.GetNetworkOutputBindingInfo(requestedOutputs.at(0));
            armnn::IOptimizedNetworkPtr optNet{nullptr, [](armnn::IOptimizedNetwork *){}};
            {
                optNet = armnn::Optimize(*network, m_Runtime->GetDeviceSpec());
            }
	    armnn::Status ret = m_Runtime->LoadNetwork(m_NetworkIdentifier, std::move(optNet));
            if (ret == armnn::Status::Failure)
            {
                throw armnn::Exception("IRuntime::LoadNetwork failed");
            }
	}
	
        void ArmnnInference::Run() {
#ifdef ARM_DEBUG
	    std::cout << "ArmnnInference::Run::start" << std::endl;
#endif
            armnn::Status ret = m_Runtime->EnqueueWorkload(m_NetworkIdentifier,
                                                           MakeInputTensors(m_Input),
                                                           MakeOutputTensors(m_Output));
#ifdef ARM_DEBUG
	    std::cout << "ArmnnInference::Run::done" << std::endl;          
#endif
            if (ret == armnn::Status::Failure)
            {
                throw armnn::Exception("IRuntime::EnqueueWorkload failed");
            }
        }

    } // namespace armnn_backend
} // namespace menoh_impl
