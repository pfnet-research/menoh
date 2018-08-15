#pragma once

#ifndef MENOH_PARSER_HPP
#define MENOH_PARSER_HPP

#include "armnn/ArmNN.hpp"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include <iostream>

#include <menoh/array.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>

using namespace armnn;

namespace menoh_impl {
    namespace armnn_backend {

        using BindingPointInfo = std::pair<armnn::LayerBindingId, armnn::TensorInfo>;

        class MenohParser;

	/// An Abstract base class which represents a single tensorflow operation (node)
        /// that has been (potentially partially) converted to Armnn.
        /// It may not yet have been fully converted into actual Armnn layers.
        class ParsedMenohOperation {
        public:
            ParsedMenohOperation(MenohParser* parser, const node& node)
            : m_Parser(parser)
            , m_Node(node){
            }

            virtual ~ParsedMenohOperation() {};

            const node& GetNode() const { return m_Node; }

            /// Gets the ArmNN IOutputSlot corresponding to the given output index of the Tensorflow operation.
            /// This may result in the creation of Armnn layers if this was deferred (e.g. see ParsedConstMenohOperation).
            virtual IOutputSlot& ResolveArmnnOutputSlot(unsigned int MenohOutputIndex) = 0;

            /// If this operation is an Identity then this will follow return the 'parent' operation (recursively).
            virtual ParsedMenohOperation* ResolveIdentityOperations(){
                return this;
            }

        protected:
            MenohParser* m_Parser;
            const node& m_Node;
        };

        using ParsedMenohOperationPtr = std::unique_ptr<ParsedMenohOperation>;
        
        ///
        /// WithOutputTensorIndex wraps a value and an index. The purpose of
        /// this template is to signify that in Tensorflow the input name of
        /// a layer has the convention of 'inputTensorName:#index' where the
        /// #index can be omitted and it implicitly means the 0. output of
        /// the referenced layer. By supporting this notation we can handle
        /// layers with multiple outputs, such as Split.
        ///
        template <typename T>
        struct WithOutputTensorIndex
        {
            T                m_IndexedValue;
            unsigned int     m_Index;

            WithOutputTensorIndex(const T & value, unsigned int index)
            : m_IndexedValue{value}
            , m_Index{index} {}

            WithOutputTensorIndex(T && value, unsigned int index)
            : m_IndexedValue{value}
            , m_Index{index} {}
        };

        using OutputOfParsedMenohOperation = WithOutputTensorIndex<ParsedMenohOperation *>;
        using OutputOfConstNodeDef = WithOutputTensorIndex<const node*>;
        using OutputId = WithOutputTensorIndex<std::string>;

        class MenohParser {

        public:
            MenohParser();
            armnn::INetworkPtr CreateNetworkFromGraph(const graph& menoh_graph,
		 std::unordered_map<std::string, array> const& parameter_table,
	         const std::map<std::string, armnn::TensorShape>& inputShapes,
                 const std::vector<std::string>& requestedOutputs);

            BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const;
            BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const;

        private:
            template <typename T>
            friend class ParsedConstMenohOperation;
            friend class ParsedMatMulMenohOperation;

            /// Parses a Menoh Graph loaded into memory from one of the other CreateNetwork*

            /// sets up variables and then performs BFS to parse all nodes
            void LoadGraph(const graph& menoh_graph,
                          std::unordered_map<std::string, array> const& parameter_table);
      
            /// parses a given node, assuming nodes before it in graph have been done
  	    void LoadNode(const node& menoh_node, const graph& menoh_graph);

            /// Handling identity layers as the input for Conv2D layer
            const node* ResolveIdentityNode(const node* menoh_node);
            /// Finds the nodes connected as inputs of the given node in the graph.
            std::vector<OutputOfConstNodeDef> GetMenohInputNodes(const node& menoh_node) const;
            /// Finds the IParsedMenohOperations for the nodes connected as inputs of the given node in the graph,
            /// and throws an exception if the number of inputs does not match the expected one.
            /// This will automatically resolve any identity nodes. The result vector contains the parsed operation
            /// together with the output tensor index to make the connection unambiguous.
            std::vector<OutputOfParsedMenohOperation> GetInputParsedMenohOperationsChecked(const node& menoh_node,
                                                                                    std::size_t expectedNumInputs);

	  //            ParsedMenohOperationPtr ParseConst(const node& menoh_node, const graph& menoh_graph);

            /// Checks if there is a pre-parsed const tensor is available with the given name and Type
            template<typename Type>
            bool HasParsedConstTensor(const std::string & nodeName) const;

            ParsedMenohOperationPtr ParseConst(           const menoh_impl::node& node, const menoh_impl::graph& graph);
	    ParsedMenohOperationPtr ParseAdd(             const menoh_impl::node& node, const menoh_impl::graph& graph);
	    ParsedMenohOperationPtr ParseBiasAdd(         const menoh_impl::node& node, const menoh_impl::graph& graph);
	    ParsedMenohOperationPtr ParseFC(              const menoh_impl::node& node, const menoh_impl::graph& graph);
	    ParsedMenohOperationPtr ParseConv2D(          const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseDepthwiseConv2D( const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseFusedBatchNorm(  const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseConcat(          const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseIdentity(        const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseLrn(             const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseMatMul(          const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseMul(             const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParsePlaceholder(     const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseRelu(            const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseRelu6(           const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseReshape(         const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseResizeBilinear(  const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseShape(           const menoh_impl::node& node, const menoh_impl::graph& graph);
//            ParsedMenohOperationPtr ParseSqueeze(         const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseSigmoid(         const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseSoftmax(         const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseSoftplus(        const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseTanh(            const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseMaxPool(         const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseAvgPool(         const menoh_impl::node& node, const menoh_impl::graph& graph);

            ParsedMenohOperationPtr ParsePooling2d(       const menoh_impl::node& node, const menoh_impl::graph& graph,
                                                          armnn::PoolingAlgorithm pooltype);
            ParsedMenohOperationPtr AddActivationLayer(   const menoh_impl::node& node, armnn::ActivationDescriptor& desc);
            ParsedMenohOperationPtr AddAdditionLayer(     const menoh_impl::node& node, bool isBiasAdd = false);

            IConnectableLayer* AddFullyConnectedLayer(    const menoh_impl::node& matMulNodeDef, 
                                                          const menoh_impl::node* addNodeDef, const char* armnnLayerName);
 	    IConnectableLayer* AddFullyConnectedLayer(    const menoh_impl::node& node, const char* armnnLayerName);
	  
            static std::pair<armnn::LayerBindingId, armnn::TensorInfo> GetBindingInfo(const std::string& layerName,
                const char* bindingPointDesc,
                const std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo);

            void TrackInputBinding(armnn::IConnectableLayer* layer,
                armnn::LayerBindingId id,
                const armnn::TensorInfo& tensorInfo);

            void TrackOutputBinding(armnn::IConnectableLayer* layer,
                armnn::LayerBindingId id,
                const armnn::TensorInfo& tensorInfo);

            static void TrackBindingPoint(armnn::IConnectableLayer* layer, armnn::LayerBindingId id,
                const armnn::TensorInfo& tensorInfo,
                const char* bindingPointDesc,
                std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo);

            void Cleanup();

            /// The network we're building. Gets cleared after it is passed to the user
            armnn::INetworkPtr m_Network;

            using OperationParsingFunction = ParsedMenohOperationPtr(MenohParser::*)(
						                     const menoh_impl::node& node, const menoh_impl::graph& graph);

            /// map of TensorFlow operation names to parsing member functions
            static const std::map<std::string, OperationParsingFunction> ms_OperationNameToParsingFunctions;

            std::map<std::string, armnn::TensorShape> m_InputShapes;
            std::vector<std::string> m_RequestedOutputs;

            /// map of nodes extracted from the Graph to speed up parsing
            std::unordered_map<std::string, const node*> m_NodesByName;
            std::unordered_map<std::string, array> m_ParamByName;

            std::unordered_map<std::string, ParsedMenohOperationPtr> m_ParsedMenohOperations;

            /// maps input layer names to their corresponding ids and tensor infos
            std::unordered_map<std::string, BindingPointInfo> m_NetworkInputsBindingInfo;

            /// maps output layer names to their corresponding ids and tensor infos
            std::unordered_map<std::string, BindingPointInfo> m_NetworkOutputsBindingInfo;                
        };
    
    } // namespace armnn_backend
} // namespace menoh_impl
#endif // MENOH_PARSER_HPP
