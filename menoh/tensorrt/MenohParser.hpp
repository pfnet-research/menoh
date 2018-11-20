#ifndef MENOH_PARSER_HPP
#define MENOH_PARSER_HPP

#include <map>

#include <NvInfer.h>

#include <menoh/array.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>

#include "Tensor.hpp"

using namespace nvinfer1;

namespace menoh_impl {
    namespace tensorrt_backend {

        using BindingPointInfo = std::pair<int, TensorInfo>;

        using LayerBindingId = int;
        
        class MenohParser;

        class ParsedMenohOperation {
        public:
            ParsedMenohOperation(MenohParser* parser, const node& node)
            : m_Parser(parser)
            , m_Node(node){}

            virtual ~ParsedMenohOperation() {};

            const node& GetNode() const { return m_Node; }

            virtual ITensor* ResolveTensorRTOutputSlot(unsigned int MenohOutputIndex) = 0;

            virtual ParsedMenohOperation* ResolveIdentityOperations(){
                return this;
            }

        protected:
            MenohParser* m_Parser;
            const node& m_Node;
        };

        using ParsedMenohOperationPtr = std::unique_ptr<ParsedMenohOperation>;
        
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

            INetworkDefinition* CreateNetworkFromGraph(
                                IBuilder* builder,
                                const graph& menoh_graph,
                                std::unordered_map<std::string, array> const& parameter_table,
                                const std::map<std::string, TensorShape>& inputShapes,
                                const std::vector<std::string>& requestedOutputs);

        private:
            template <typename T>
            friend class ParsedConstMenohOperation;
            friend class ParsedMatMulMenohOperation;

            void SetLayer(ILayer* layer);
            
            void LoadGraph(const graph& menoh_graph,
                          std::unordered_map<std::string, array> const& parameter_table);
      
            void LoadNode(const node& menoh_node, const graph& menoh_graph);

            const node* ResolveIdentityNode(const node* menoh_node);
            std::vector<OutputOfConstNodeDef> GetMenohInputNodes(const node& menoh_node) const;
            std::vector<OutputOfParsedMenohOperation> GetInputParsedMenohOperationsChecked(const node& menoh_node,
                                                                                    std::size_t expectedNumInputs);

            template<typename Type>
            bool HasParsedConstTensor(const std::string & nodeName) const;

            ParsedMenohOperationPtr ParseConst(             const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseBatchNormalization(const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseFC(                const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseGemm(              const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseUnsqueeze(         const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseConv2D(            const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseConcat(            const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseIdentity(          const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseLrn(               const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseMatMul(            const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseMul(               const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseAdd(               const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseSum(               const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParsePlaceholder(       const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseRelu(              const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseSigmoid(           const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseSoftmax(           const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseTanh(              const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseMaxPool(           const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParseAvgPool(           const menoh_impl::node& node, const menoh_impl::graph& graph);
            ParsedMenohOperationPtr ParsePooling2d(         const menoh_impl::node& node, const menoh_impl::graph& graph,
                                                            PoolingType pooltype);

            ParsedMenohOperationPtr AddActivationLayer(   const menoh_impl::node& node, ActivationType activationType);
	  
            ILayer* AddFullyConnectedLayer(    const menoh_impl::node& matMulNodeDef, 
                                               const menoh_impl::node* addNodeDef, const char* armnnLayerName);
            ILayer* AddFullyConnectedLayer(    const menoh_impl::node& node, const char* armnnLayerName);

            void Cleanup();

            INetworkDefinition* m_Network;

            using OperationParsingFunction = ParsedMenohOperationPtr(MenohParser::*)(
						                     const menoh_impl::node& node, const menoh_impl::graph& graph);

            static const std::map<std::string, OperationParsingFunction> ms_OperationNameToParsingFunctions;

            std::map<std::string, TensorShape> m_InputShapes;
            std::vector<std::string> m_RequestedOutputs;
            ILayer* m_Layer;
            std::map<const char*, ILayer*> m_LayerMap;

            std::unordered_map<std::string, const node*> m_NodesByName;
            std::unordered_map<std::string, array> m_ParamByName;

            std::unordered_map<std::string, ParsedMenohOperationPtr> m_ParsedMenohOperations;

            std::unordered_map<std::string, BindingPointInfo> m_NetworkInputsBindingInfo;

            std::unordered_map<std::string, BindingPointInfo> m_NetworkOutputsBindingInfo;                
        };
    
    } // namespace tensorrt_backend
} // namespace menoh_impl
#endif // MENOH_PARSER_HPP
