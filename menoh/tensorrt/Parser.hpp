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

        constexpr unsigned int GetDataTypeSize(DataType dataType) {
            switch(dataType) {
                case DataType::kINT32:
                case DataType::kFLOAT:
                    return 4U;
                default:
                    return 0U;
            }
        }

        class Parser;

        class Operation {
        public:
            Operation(Parser* parser, const node& node)
              : m_Parser(parser), m_Node(node) {}

            virtual ~Operation(){};

            const node& GetNode() const { return m_Node; }

            virtual ITensor* Output(unsigned int index) = 0;

            virtual Operation* IdentityOperations() { return this; }

        protected:
            Parser* m_Parser;
            const node& m_Node;
        };

        using OperationPtr = std::unique_ptr<Operation>;

        template <typename T>
        struct WithOutputTensorIndex {
            T m_Value;
            unsigned int m_Index;

            WithOutputTensorIndex(const T& value, unsigned int index)
              : m_Value{value}, m_Index{index} {}

            WithOutputTensorIndex(T&& value, unsigned int index)
              : m_Value{value}, m_Index{index} {}
        };

        using OutputOfOperation = WithOutputTensorIndex<Operation*>;
        using OutputOfConstNodeDef = WithOutputTensorIndex<const node*>;

        class Parser {

        public:
            Parser() : m_Network() {}

            INetworkDefinition* CreateNetwork(
              IBuilder* builder, const graph& menoh_graph,
              std::unordered_map<std::string, array> const& parameter_table,
              const std::vector<std::string>& outputs);

            INetworkDefinition* Network();

            std::string ConvertToInputTensorName(std::string const& name);
            std::string ConvertToOutputTensorName(std::string const& name);

        private:
            template <typename T>
            friend class ConstOperation;

            void InitLayerNameAndOutputTensorName(ILayer* layer,
                                                  const menoh_impl::node& node);
            void InitLayerNameAndOutputTensorName(ILayer* layer,
                                                  const menoh_impl::node& node,
                                                  int index);

            void MarkOutputIfRequired(
              ITensor* output_tensor, std::string const& output_name,
              std::vector<std::string> const& required_outputs);

            void CheckOutput(const menoh_impl::graph& graph,
                             const std::vector<std::string>& outputs);
            void LoadParameter(
              std::unordered_map<std::string, array> const& parameter_table);
            void LoadNode(const menoh_impl::node& node,
                          std::vector<std::string> const& required_outputs);
            void LoadGraph(const menoh_impl::graph& graph,
                           std::vector<std::string> const& required_outputs);

            std::vector<OutputOfConstNodeDef>
            InputNodes(const menoh_impl::node& node) const;
            std::vector<OutputOfOperation>
            InputCheck(const menoh_impl::node& node,
                       std::size_t expectedNumInputs);
            ITensor* GetTensor(OutputOfOperation& input);

            template <typename Type>
            bool HasParsedConst(const std::string& nodeName) const;
            bool HasParsedConst(OutputOfOperation& input);

            OperationPtr
            ParseConst(const menoh_impl::node& node,
                       std::vector<std::string> const& required_outputs);
            OperationPtr ParseBatchNormalization(
              const menoh_impl::node& node,
              std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseFC(const menoh_impl::node& node,
                    std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseGemm(const menoh_impl::node& node,
                      std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseUnsqueeze(const menoh_impl::node& node,
                           std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseConv2D(const menoh_impl::node& node,
                        std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseConcat(const menoh_impl::node& node,
                        std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseIdentity(const menoh_impl::node& node,
                          std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseLrn(const menoh_impl::node& node,
                     std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseMul(const menoh_impl::node& node,
                     std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseAdd(const menoh_impl::node& node,
                     std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseSum(const menoh_impl::node& node,
                     std::vector<std::string> const& required_outputs);
            OperationPtr
            ParsePlaceholder(const menoh_impl::node& node,
                             std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseRelu(const menoh_impl::node& node,
                      std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseSigmoid(const menoh_impl::node& node,
                         std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseSoftmax(const menoh_impl::node& node,
                         std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseTanh(const menoh_impl::node& node,
                      std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseMaxPool(const menoh_impl::node& node,
                         std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseAvgPool(const menoh_impl::node& node,
                         std::vector<std::string> const& required_outputs);
            OperationPtr ParseGlobalMaxPool(
              const menoh_impl::node& node,
              std::vector<std::string> const& required_outputs);
            OperationPtr ParseGlobalAvgPool(
              const menoh_impl::node& node,
              std::vector<std::string> const& required_outputs);

            OperationPtr
            ParseElementWise(const menoh_impl::node& node,
                             ElementWiseOperation op,
                             std::vector<std::string> const& required_outputs);
            OperationPtr
            ParseGlobalPool(const menoh_impl::node& node, PoolingType type,
                            std::vector<std::string> const& required_outputs);
            OperationPtr AddActivationLayer(
              const menoh_impl::node& node, ActivationType activationType,
              std::vector<std::string> const& required_outputs);

            void Cleanup();

            using ParseFunction = OperationPtr (Parser::*)(
              const menoh_impl::node& node,
              std::vector<std::string> const& required_outputs);

            static const std::map<std::string, ParseFunction> m_Functions;

            INetworkDefinition* m_Network;

            std::unordered_map<std::string, const node*> m_Nodes;
            std::unordered_map<std::string, array> m_Params;
            std::unordered_map<std::string, OperationPtr> m_Operations;

            std::unordered_map<std::string, std::string>
              input_tensor_name_table;
            std::unordered_map<std::string, std::string>
              output_tensor_name_table;
        };

    } // namespace tensorrt_backend
} // namespace menoh_impl
#endif // MENOH_PARSER_HPP
