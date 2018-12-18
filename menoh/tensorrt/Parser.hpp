#ifndef MENOH_PARSER_HPP
#define MENOH_PARSER_HPP

#include <map>

#include <NvInfer.h>

#include <menoh/array.hpp>
#include <menoh/model_core.hpp>
#include <menoh/model_data.hpp>

using namespace nvinfer1;

namespace menoh_impl {
    namespace tensorrt_backend {

        constexpr unsigned int GetDataTypeSize(DataType dataType)
        {
            switch (dataType)
            {
                case DataType::kINT32:
                case DataType::kFLOAT:     return 4U;
                default:                   return 0U;
            }
        }

        class Parser;

        class Operation {
        public:
            Operation(Parser* parser, const node& node)
            : m_Parser(parser)
            , m_Node(node){}

            virtual ~Operation() {};

            const node& GetNode() const { return m_Node; }

            virtual ITensor* Output(unsigned int index) = 0;

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

        class Parser {

        public:
            Parser()
              : m_Network()
              , m_Layer()
            {}

            INetworkDefinition* CreateNetwork(
                                IBuilder* builder,
                                const graph& menoh_graph,
                                std::unordered_map<std::string, array> const& parameter_table,
                                const std::vector<std::string>& outputs);

            INetworkDefinition* Network();
          
        private:
            template <typename T>
            friend class ConstOperation;

            void SetLayer(ILayer* layer, const menoh_impl::node &node, const char* index = "" );
            
            void CheckOutput(const menoh_impl::graph& graph, const std::vector<std::string>& outputs);
            void LoadParameter(std::unordered_map<std::string, array> const& parameter_table);
            void LoadNode(const menoh_impl::node& node);
            void LoadGraph(const menoh_impl::graph& graph);

            std::vector<OutputOfConstNodeDef> InputNodes(const menoh_impl::node& node) const;
            std::vector<OutputOfOperation> InputCheck(const menoh_impl::node& node, std::size_t expectedNumInputs);
            ITensor* GetTensor(OutputOfOperation& input);
          
            template<typename Type>
            bool HasParsedConst(const std::string& nodeName) const;
            bool HasParsedConst(OutputOfOperation& input);
          
            OperationPtr ParseConst(             const menoh_impl::node& node);
            OperationPtr ParseBatchNormalization(const menoh_impl::node& node);
            OperationPtr ParseFC(                const menoh_impl::node& node);
            OperationPtr ParseGemm(              const menoh_impl::node& node);
            OperationPtr ParseUnsqueeze(         const menoh_impl::node& node);
            OperationPtr ParseConv2D(            const menoh_impl::node& node);
            OperationPtr ParseConcat(            const menoh_impl::node& node);
            OperationPtr ParseIdentity(          const menoh_impl::node& node);
            OperationPtr ParseLrn(               const menoh_impl::node& node);
            OperationPtr ParseMul(               const menoh_impl::node& node);
            OperationPtr ParseAdd(               const menoh_impl::node& node);
            OperationPtr ParseSum(               const menoh_impl::node& node);
            OperationPtr ParsePlaceholder(       const menoh_impl::node& node);
            OperationPtr ParseRelu(              const menoh_impl::node& node);
            OperationPtr ParseSigmoid(           const menoh_impl::node& node);
            OperationPtr ParseSoftmax(           const menoh_impl::node& node);
            OperationPtr ParseTanh(              const menoh_impl::node& node);
            OperationPtr ParseMaxPool(           const menoh_impl::node& node);
            OperationPtr ParseAvgPool(           const menoh_impl::node& node);
            OperationPtr ParseGlobalMaxPool(     const menoh_impl::node& node);
            OperationPtr ParseGlobalAvgPool(     const menoh_impl::node& node);

            OperationPtr ParseElementWise(       const menoh_impl::node& node, ElementWiseOperation op);
            OperationPtr ParseGlobalPool(        const menoh_impl::node& node, PoolingType type);
            OperationPtr AddActivationLayer(     const menoh_impl::node& node, ActivationType activationType);

            void Cleanup();

            using ParseFunction = OperationPtr(Parser::*)(const menoh_impl::node& node);

            static const std::map<std::string, ParseFunction> m_Functions;

            INetworkDefinition*                           m_Network;
            ILayer*                                       m_Layer;

            std::unordered_map<std::string, const node*>  m_Nodes;
            std::unordered_map<std::string, array>        m_Params;
            std::unordered_map<std::string, OperationPtr> m_Operations;
        };
    
    } // namespace tensorrt_backend
} // namespace menoh_impl
#endif // MENOH_PARSER_HPP
