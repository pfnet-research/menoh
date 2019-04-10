#ifndef MENOH_PARSER_HPP
#define MENOH_PARSER_HPP

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include <iostream>

#include <armnn/ArmNN.hpp>

#include <menoh/array.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>

using namespace armnn;

namespace menoh_impl {
    namespace armnn_backend {

        using BindingPointInfo = std::pair<armnn::LayerBindingId, TensorInfo>;

        class Parser;

        class Operation {
        public:
            Operation(Parser* parser, const node& node)
            : m_Parser(parser)
            , m_Node(node){
            }

            virtual ~Operation() {};

            const node& GetNode() const { return m_Node; }

            virtual IOutputSlot& Output(unsigned int MenohOutputIndex) = 0;

            virtual Operation* IdentityOperations(){
                return this;
            }

        protected:
            Parser* m_Parser;
            const node& m_Node;
        };

        using OperationPtr = std::unique_ptr<Operation>;
        
        template <typename T>
        struct WithOutputTensorIndex
        {
            T                m_Value;
            unsigned int     m_Index;

            WithOutputTensorIndex(const T & value, unsigned int index)
            : m_Value{value}
            , m_Index{index} {}

            WithOutputTensorIndex(T && value, unsigned int index)
            : m_Value{value}
            , m_Index{index} {}
        };

        using OutputOfOperation = WithOutputTensorIndex<Operation *>;
        using OutputOfConstNodeDef = WithOutputTensorIndex<const node*>;
        using OutputId = WithOutputTensorIndex<std::string>;

        class Parser {

        public:
            Parser();

            armnn::INetworkPtr CreateNetworkFromGraph(const graph& menoh_graph,
		 std::unordered_map<std::string, array> const& parameter_table,
                 const std::vector<std::string>& outputs);

            BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const;
            BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const;

        private:
            template <typename T>
            friend class ConstOperation;
            friend class ParsedMatMulMenohOperation;

            void CheckOutput(const menoh_impl::graph& graph, const std::vector<std::string>& outputs);
            void LoadParameter(std::unordered_map<std::string, array> const& parameter_table);
            void LoadNode(const menoh_impl::node& node);
            void LoadGraph(const menoh_impl::graph& graph);
      
            const node* IdentityNode(const node* menoh_node);

            std::vector<OutputOfConstNodeDef> InputNodes(const menoh_impl::node& node) const;
            std::vector<OutputOfOperation> InputCheck(const menoh_impl::node& node, std::size_t expectedNumInputs);
          //            ITensor* GetTensor(OutputOfOperation& input);

            template<typename Type>
            bool HasParsedConst(const std::string & nodeName) const;
            bool HasParsedConst(OutputOfOperation& input);
            IOutputSlot* GetSlot(OutputOfOperation& input);
          
            OperationPtr ParseConst(             const menoh_impl::node& node);
            OperationPtr ParseAdd(               const menoh_impl::node& node);
            OperationPtr ParseBiasAdd(           const menoh_impl::node& node);
            OperationPtr ParseFC(                const menoh_impl::node& node);
            OperationPtr ParseGemm(              const menoh_impl::node& node);
            OperationPtr ParseConv2D(            const menoh_impl::node& node);
            OperationPtr ParseDepthwiseConv2D(   const menoh_impl::node& node);
            OperationPtr ParseBatchNormalization(const menoh_impl::node& node);
            OperationPtr ParseConcat(            const menoh_impl::node& node);
            OperationPtr ParseConcatV2(          const menoh_impl::node& node);
            OperationPtr ParseIdentity(          const menoh_impl::node& node);
            OperationPtr ParseLrn(               const menoh_impl::node& node);
            OperationPtr ParseMatMul(            const menoh_impl::node& node);
            OperationPtr ParseMul(               const menoh_impl::node& node);
            OperationPtr ParsePlaceholder(       const menoh_impl::node& node);
            OperationPtr ParseRelu(              const menoh_impl::node& node);
            OperationPtr ParseRelu6(             const menoh_impl::node& node);
            OperationPtr ParseReshape(           const menoh_impl::node& node);
            OperationPtr ParseResizeBilinear(    const menoh_impl::node& node);
            OperationPtr ParseShape(             const menoh_impl::node& node);
            OperationPtr ParseSigmoid(           const menoh_impl::node& node);
            OperationPtr ParseSoftmax(           const menoh_impl::node& node);
            OperationPtr ParseSoftplus(          const menoh_impl::node& node);
            OperationPtr ParseTanh(              const menoh_impl::node& node);
            OperationPtr ParseMaxPool(           const menoh_impl::node& node);
            OperationPtr ParseAvgPool(           const menoh_impl::node& node);
            OperationPtr ParsePooling2d(         const menoh_impl::node& node, armnn::PoolingAlgorithm pooltype);
            OperationPtr ParseGlobalMaxPool(     const menoh_impl::node& node);
            OperationPtr ParseGlobalAvgPool(     const menoh_impl::node& node);
            OperationPtr ParseGlobalPooling2d(   const menoh_impl::node& node, armnn::PoolingAlgorithm pooltype);
          
            OperationPtr AddActivationLayer(     const menoh_impl::node& node, armnn::ActivationDescriptor& desc);
            OperationPtr AddAdditionLayer(       const menoh_impl::node& node, bool isBiasAdd = false);

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

            using ParseFunction = OperationPtr(Parser::*)(const menoh_impl::node& node);

            static const std::map<std::string, ParseFunction> m_Functions;

            armnn::INetworkPtr m_Network;

            std::vector<std::string> m_Outputs;

            std::unordered_map<std::string, const node*> m_Nodes;
            std::unordered_map<std::string, array> m_Params;

            std::unordered_map<std::string, OperationPtr> m_Operations;

            std::unordered_map<std::string, BindingPointInfo> m_NetworkInputsBindingInfo;

            std::unordered_map<std::string, BindingPointInfo> m_NetworkOutputsBindingInfo;                
        };
    
    } // namespace armnn_backend
} // namespace menoh_impl
#endif // PARSER_HPP
